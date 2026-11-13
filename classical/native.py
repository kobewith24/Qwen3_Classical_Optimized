"""Dedicated authenticated llama.cpp slot; real SSE, usage and cancellation fencing."""
import http.client
import json
import queue
import socket
import threading
import time
from .constraints import POLICY_VERSION,compile_punctuation
from .core import ROOT,digest,messages


class TextStream:
    def __init__(self):self.queue=queue.Queue()
    def __iter__(self):return self
    def __next__(self):
        value=self.queue.get(timeout=.2)
        if value is None:raise StopIteration
        return value
    def put(self,text):self.queue.put(text)
    def end(self):self.queue.put(None)


class NativeEngine:
    def __init__(self,port=11436,constrain=True):
        from transformers import AutoTokenizer
        self.port=port;self.constrain=constrain;self.lock=threading.Lock();self.quarantined=False
        self.token=(ROOT/'data/native.token').read_text(encoding='utf-8-sig').strip()
        state=json.loads((ROOT/'data/native-process.json').read_text(encoding='utf-8-sig'))
        if state['port']!=port or state['root']!=str(ROOT):raise ValueError('原生服务启动记录不匹配。')
        self.model_name=state['model'];self.model_sha256=state['model_sha256'];self.binary_sha256=state['binary_sha256']
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model',local_files_only=True)
        props=self.control('GET','/props');slots=self.control('GET','/slots')
        if props.get('total_slots')!=1 or len(slots)!=1:raise ValueError('原生后端必须独占一个生成槽位。')
        if props.get('model_path') and str(props['model_path']).replace('\\','/')!=state['model_path'].replace('\\','/'):
            raise ValueError('原生服务模型路径不匹配。')
        self.verify_tokenizer()

    def control(self,method,path,payload=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        try:
            conn.request(method,path,body=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None,
                         headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
            response=conn.getresponse();data=response.read()
            if response.status!=200:raise RuntimeError(f'Native endpoint {path}: HTTP {response.status}')
            return json.loads(data)
        finally:conn.close()

    def verify_tokenizer(self):
        sample=[{'role':'system','content':'准确作答。'},{'role':'user','content':'𠀀春眠不觉晓。'}]
        text=self.tokenizer.apply_chat_template(sample,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        expected=self.tokenizer.encode(text,add_special_tokens=False)
        actual=self.control('POST','/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens']
        if expected!=actual:raise ValueError('GGUF tokenizer 与本地 HF 提示模板不一致。')

    def fingerprint(self):
        return digest({'backend':'llama.cpp','model':self.model_sha256,'binary':self.binary_sha256,
                       'template':self.tokenizer.chat_template,'grammar':POLICY_VERSION if self.constrain else None,
                       'context':2048,'cache_prompt':False,'temperature':0,'seed':24})

    def make_streamer(self):return TextStream()

    def wait_idle(self,seconds=15):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            try:
                if all(not row['is_processing'] for row in self.control('GET','/slots')):return
            except (OSError,RuntimeError):pass
            time.sleep(.1)
        self.quarantined=True
        raise RuntimeError('无法确认原生槽位已释放；后端已隔离，请重启服务。')

    def generate_reader(self,item,**kwargs):
        constraint=compile_punctuation(item['input']) if self.constrain and item['task']=='punctuate' else None
        out=self.generate_messages(messages(item),constraint=constraint,**kwargs)
        if constraint:
            out['constraint']={'policy':POLICY_VERSION,'fingerprint':constraint.fingerprint,
                               'complete':out['finish_reason']=='stop' and constraint.accepts(out['text'])}
            if out['finish_reason']=='stop' and not out['constraint']['complete']:
                raise ValueError('原生输出未通过正文约束校验。')
        return out

    def generate_messages(self,turns,max_new_tokens=256,cancel=None,timeout=120,streamer=None,constraint=None):
        cancel=cancel or threading.Event()
        with self.lock:
            if self.quarantined:raise RuntimeError('原生后端已隔离，请重启服务。')
            if cancel.is_set():raise InterruptedError()
            prompt=self.tokenizer.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=self.tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)>1024:raise ValueError('Input exceeds 1024 tokens; split the source text')
            payload={'prompt':ids,'n_predict':max_new_tokens,'stream':True,'temperature':0,'seed':24,
                     'repeat_penalty':1.,'cache_prompt':False,'id_slot':0,'return_tokens':True}
            if constraint:payload['grammar']=constraint.grammar
            conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=timeout+5)
            conn.connect();transport=conn.sock
            done=threading.Event();expired=threading.Event();start=time.perf_counter();deadline=time.monotonic()+timeout
            def interrupt():
                while not done.wait(.02):
                    if cancel.is_set() or time.monotonic()>=deadline:
                        if not cancel.is_set():expired.set()
                        if transport is not None:
                            try:transport.shutdown(socket.SHUT_RDWR)
                            except OSError:pass
                        return
            watcher=threading.Thread(target=interrupt,daemon=True);watcher.start();pieces=[];final=None
            try:
                conn.request('POST','/completion',body=json.dumps(payload,ensure_ascii=False).encode(),
                    headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
                response=conn.getresponse()
                if response.status!=200:raise RuntimeError(f'Native generation HTTP {response.status}')
                while True:
                    line=response.readline()
                    if not line:break
                    if not line.startswith(b'data: '):continue
                    data=json.loads(line[6:])
                    if data.get('error'):raise RuntimeError('原生推理未完成。')
                    if data.get('content'):
                        pieces.append(data['content'])
                        if streamer:streamer.put(data['content'])
                    if data.get('stop'):final=data;break
                if cancel.is_set():raise InterruptedError('Generation cancelled')
                if expired.is_set():raise TimeoutError('Generation exceeded deadline')
                if final is None:raise RuntimeError('原生 SSE 提前结束，缺少最终 usage。')
                if final.get('truncated'):raise ValueError('原生上下文发生截断，拒绝返回部分结果。')
                timings=final.get('timings',{})
                count=timings.get('predicted_n')
                if not isinstance(count,int):raise RuntimeError('原生响应缺少真实输出 token 计数。')
                # prompt_n excludes cached tokens; cache_prompt=False makes it the full prompt.
                if final.get('tokens_evaluated',timings.get('prompt_n'))!=len(ids):raise ValueError('原生输入 token 数不匹配。')
                return {'text':''.join(pieces),'completion_tokens':count,'prompt_tokens':len(ids),
                        'finish_reason':'length' if final.get('stop_type')=='limit' else 'stop',
                        'batch_seconds':time.perf_counter()-start,'native_timings':timings}
            except (OSError,http.client.HTTPException) as exc:
                if cancel.is_set():raise InterruptedError('Generation cancelled') from exc
                if expired.is_set():raise TimeoutError('Generation exceeded deadline') from exc
                raise
            finally:
                done.set();conn.close();watcher.join(timeout=1)
                try:self.wait_idle()
                finally:
                    if streamer:streamer.end()
