"""Single-process authenticated local gateway with bounded GPU work."""
import argparse
import asyncio
import contextlib
import json
import queue
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends,FastAPI,Header,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
from pydantic import BaseModel,ConfigDict,Field

from classical.core import ROOT,INSTRUCTIONS,SYSTEM,messages,write_json,digest,file_digest
from classical.reader import segments,review
from classical.documents import assemble


class Gate:
    def __init__(self,capacity=4,timeout=30):
        self.capacity=capacity;self.timeout=timeout;self.outstanding=0;self.active=0
        self.semaphore=asyncio.Semaphore(1)

    async def acquire(self):
        if self.outstanding>=self.capacity+1:raise HTTPException(429,'等待队列已满，请稍后重试。')
        self.outstanding+=1
        try:await asyncio.wait_for(self.semaphore.acquire(),self.timeout)
        except BaseException:
            self.outstanding-=1;raise
        self.active+=1

    def release(self):
        self.active-=1;self.outstanding-=1;self.semaphore.release()


class SecurityHeaders:
    """Pass receive through unchanged so disconnect signals reach request handlers."""
    def __init__(self,app):self.app=app

    async def __call__(self,scope,receive,send):
        async def wrapped(message):
            if scope['type']=='http' and message['type']=='http.response.start':
                message={**message,'headers':[*message.get('headers',[]),
                    (b'x-content-type-options',b'nosniff'),(b'referrer-policy',b'no-referrer'),
                    (b'content-security-policy',b"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'")]}
            await send(message)
        await self.app(scope,receive,wrapped)


class Message(BaseModel):
    role:str;content:str=Field(max_length=12000)


class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model:str|None=None
    messages:list[Message]=Field(min_length=1,max_length=16)
    max_tokens:int=Field(default=256,ge=1,le=640)
    temperature:float=Field(default=0,ge=0,le=0)
    stream:bool=False
    stream_options:dict|None=None


class ProcessRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    task:str
    text:str=Field(min_length=1,max_length=3000)
    request_id:str|None=Field(default=None,min_length=8,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')


class CancelRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:str=Field(min_length=8,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')


class EditRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=12000)
    revision:int=Field(ge=0)


def create_app(engine,token,model_name='classical-clean',queue_capacity=4,queue_timeout=30,generation_timeout=120,reader_task_timeout=300,store=None):
    if not token:raise ValueError('Access token is required')
    if reader_task_timeout<=0:raise ValueError('Task timeout must be positive')
    app=FastAPI(title='文言文断句与辅助阅读',version='1.0.0');gate=Gate(queue_capacity,queue_timeout)
    stats={'completed':0,'failed':0,'cancelled':0,'output_tokens':0,'cancel_requests':0,'reader_timeouts':0,'reader_cache_hits':0}
    app.state.gate=gate;app.state.engine=engine
    app.add_middleware(SecurityHeaders)
    jobs={};cancelled_before_start={};progress={};app.state.jobs=jobs;app.state.progress=progress
    leases=set()
    signature=digest({'engine':engine.fingerprint(),'prompts':[SYSTEM,INSTRUCTIONS],
        'code':{p:file_digest(ROOT/p) for p in ['api/server.py','classical/reader.py','classical/core.py','classical/constraints.py','classical/native.py','classical/documents.py']}}) if store else None

    def prepare_document(body):
        if not store:raise HTTPException(501,'当前服务未启用本地文档保存。')
        if body.task not in INSTRUCTIONS:raise HTTPException(400,'任务类型无效。')
        try:return store.prepare(body.task,body.text,signature,model_name)
        except ValueError as e:raise HTTPException(400,str(e))

    def require_document(key):
        if not re.fullmatch(r'[0-9a-f]{64}',key):raise HTTPException(400,'文档标识无效。')
        if not store:raise HTTPException(501,'当前服务未启用本地文档保存。')
        try:return store.load(key)
        except KeyError:raise HTTPException(404,'找不到本地文档。')

    def prune_cancellations():
        for key,deadline in list(cancelled_before_start.items()):
            if deadline<time.monotonic():cancelled_before_start.pop(key,None)

    async def auth(authorization:str|None=Header(default=None)):
        if not authorization or not authorization.startswith('Bearer ') or not secrets.compare_digest(authorization[7:].encode('utf-8'),token.encode('utf-8')):
            raise HTTPException(401,'访问令牌无效。')

    async def start(turns,max_tokens,cancel,streamer=None,reader_item=None):
        acquisition=asyncio.create_task(gate.acquire())
        try:
            while not acquisition.done():
                if cancel.is_set():raise HTTPException(409,'处理已停止，原文保留。')
                await asyncio.wait({acquisition},timeout=.05)
            try:await acquisition
            except asyncio.TimeoutError:raise HTTPException(429,'排队等待超时。')
            if cancel.is_set():raise HTTPException(409,'处理已停止，原文保留。')
            generate=engine.generate_reader if reader_item is not None and hasattr(engine,'generate_reader') else engine.generate_messages
            argument=reader_item if generate==getattr(engine,'generate_reader',None) else turns
            task=asyncio.create_task(asyncio.to_thread(generate,argument,max_new_tokens=max_tokens,
                                                     cancel=cancel,timeout=generation_timeout,streamer=streamer))
        except BaseException:
            if not acquisition.done():
                acquisition.cancel();await asyncio.gather(acquisition,return_exceptions=True)
            if not acquisition.cancelled() and acquisition.exception() is None:gate.release()
            raise
        def finished(future):
            gate.release()
            if future.cancelled():stats['cancelled']+=1;return
            error=future.exception()
            if error:
                stats['cancelled' if isinstance(error,InterruptedError) else 'failed']+=1
            else:
                stats['completed']+=1;stats['output_tokens']+=future.result()['completion_tokens']
        task.add_done_callback(finished)
        return task

    async def await_result(turns,max_tokens,request,cancel=None,reader_item=None):
        cancel=cancel or threading.Event()
        async def watch_disconnect():
            while True:
                if (await request.receive())['type']=='http.disconnect':cancel.set();return
        monitor=asyncio.create_task(watch_disconnect())
        try:
            task=await start(turns,max_tokens,cancel,reader_item=reader_item)
            while not task.done():
                if cancel.is_set():raise HTTPException(409,'处理已停止，原文保留。')
                await asyncio.wait({task},timeout=.15)
            if cancel.is_set():raise HTTPException(409,'处理已停止，原文保留。')
            return await asyncio.shield(task)
        except asyncio.CancelledError:cancel.set();raise
        except InterruptedError:raise HTTPException(409,'处理已停止，原文保留。')
        except TimeoutError:raise HTTPException(504,'生成超过总时间上限。')
        except ValueError as e:raise HTTPException(400,str(e))
        except RuntimeError as e:raise HTTPException(503,str(e))
        finally:
            monitor.cancel();await asyncio.gather(monitor,return_exceptions=True)

    @app.get('/')
    async def index():return FileResponse(ROOT/'api/static/index.html')

    @app.get('/app.js')
    async def javascript():return FileResponse(ROOT/'api/static/app.js',media_type='text/javascript')

    @app.get('/app.css')
    async def stylesheet():return FileResponse(ROOT/'api/static/app.css',media_type='text/css')

    @app.get('/health')
    async def health():return {'status':'unavailable' if getattr(engine,'quarantined',False) else 'ready','model':model_name,
        'scope':'local single-worker inference','backend':getattr(engine,'backend_description','HF NF4'),
        'reader_task_timeout_seconds':reader_task_timeout}

    @app.get('/v1/models',dependencies=[Depends(auth)])
    async def models():return {'object':'list','data':[{'id':model_name,'object':'model','owned_by':'local'}]}

    @app.get('/metrics',dependencies=[Depends(auth)])
    async def metrics():return {**stats,'active':gate.active,'queued':gate.outstanding-gate.active,'queue_capacity':gate.capacity,'reader_jobs':len(jobs)}

    @app.post('/api/documents',dependencies=[Depends(auth)])
    async def create_document(body:ProcessRequest):
        key=prepare_document(body);data=store.load(key)
        return {k:data[k] for k in ['document_id','completed_segments','total_segments']}

    @app.get('/api/documents',dependencies=[Depends(auth)])
    async def recent_documents():
        if not store:raise HTTPException(501,'当前服务未启用本地文档保存。')
        return {'documents':store.recent()}

    @app.get('/api/documents/{key}',dependencies=[Depends(auth)])
    async def load_document(key:str):return require_document(key)

    @app.patch('/api/documents/{key}/segments/{index}',dependencies=[Depends(auth)])
    async def edit_document(key:str,index:int,body:EditRequest):
        require_document(key)
        if key in leases:raise HTTPException(409,'请先停止处理，再保存校对。')
        try:return store.edit(key,index,body.text,body.revision)
        except KeyError:raise HTTPException(404,'该段尚未生成。')
        except ValueError as e:raise HTTPException(400,str(e))
        except RuntimeError as e:raise HTTPException(409,str(e))

    @app.get('/api/progress/{request_id}',dependencies=[Depends(auth)])
    async def reader_progress(request_id:str):
        if not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',request_id):raise HTTPException(400,'任务标识无效。')
        row=progress.get(request_id)
        if row is None:raise HTTPException(404,'任务尚未开始或已经结束。')
        return {'request_id':request_id,'completed_segments':row['completed'],'total_segments':row['total'],
                'elapsed_seconds':round(time.monotonic()-row['started'],1),'task_timeout_seconds':reader_task_timeout}

    @app.post('/api/cancel',dependencies=[Depends(auth)])
    async def cancel_process(body:CancelRequest):
        prune_cancellations();stats['cancel_requests']+=1
        if body.request_id in jobs:jobs[body.request_id].set()
        else:
            # Browser aborts and HTTP proxies can race with original request delivery.
            if len(cancelled_before_start)>=128:cancelled_before_start.pop(next(iter(cancelled_before_start)))
            cancelled_before_start[body.request_id]=time.monotonic()+120
        return {'status':'cancel_requested','request_id':body.request_id}

    @app.post('/api/process',dependencies=[Depends(auth)])
    async def process(body:ProcessRequest,request:Request):
        if body.task not in INSTRUCTIONS:raise HTTPException(400,'只支持断句、古译今和今译古。')
        try:parts=segments(body.text)
        except ValueError as e:raise HTTPException(400,str(e))
        request_id=body.request_id or secrets.token_urlsafe(18);prune_cancellations()
        if request_id in jobs:raise HTTPException(409,'该请求正在处理。')
        if request_id in cancelled_before_start:raise HTTPException(409,'处理已停止，原文保留。')
        document_id=prepare_document(body) if store else None
        if document_id in leases:raise HTTPException(409,'该文档正在处理。')
        if document_id:leases.add(document_id)
        cancel=threading.Event();jobs[request_id]=cancel
        progress[request_id]={'completed':0,'total':len(parts),'started':time.monotonic()}
        try:
            async with asyncio.timeout(reader_task_timeout):return await process_parts(body,parts,request,cancel,request_id,document_id)
        except TimeoutError:
            cancel.set();stats['reader_timeouts']+=1
            raise HTTPException(504,'整篇处理超过时间上限，已停止；原文保留，请分批处理。')
        finally:
            jobs.pop(request_id,None);progress.pop(request_id,None)
            if document_id:leases.discard(document_id)

    async def process_parts(body,parts,request,cancel,request_id,document_id):
        rows=[];cached={r['index']:r for r in store.load(document_id)['segments'] if r.get('finish_reason','stop')=='stop' or r['status']=='user_confirmed'} if document_id else {}
        for part in parts:
            if cancel.is_set():raise HTTPException(409,'处理已停止，原文保留。')
            if part['index'] in cached:
                saved=cached[part['index']]
                rows.append({**saved,'cache_hit':True,'generation_usage':saved.get('generation_usage',saved.get('usage',{})),
                             'usage':{'prompt_tokens':0,'completion_tokens':0}});stats['reader_cache_hits']+=1
                progress[request_id]['completed']=len(rows);continue
            if not part['original'].strip():
                rows.append({**part,'candidate':part['original'],'accepted':False,'body_preserved':True,
                             'text':part['original'],'status':'unchanged','note':'空白段已保留。'})
                if document_id:rows[-1]=store.save_generated(document_id,rows[-1])
                progress[request_id]['completed']=len(rows);continue
            item={'task':body.task,'input':part['original'],'instruction':INSTRUCTIONS[body.task]}
            out=await await_result(messages(item),512 if body.task=='punctuate' else 384,request,cancel,reader_item=item)
            checked=review(body.task,part['original'],out['text'])
            if out['finish_reason']=='length':
                checked.update(status='review_required',accepted=False,note='达到输出上限，候选可能不完整，请人工复核。')
                if body.task=='punctuate':checked['text']=part['original']
            rows.append({**part,**checked,'usage':{'prompt_tokens':out['prompt_tokens'],'completion_tokens':out['completion_tokens']},
                         'generation_usage':{'prompt_tokens':out['prompt_tokens'],'completion_tokens':out['completion_tokens']},
                         'cache_hit':False,'finish_reason':out['finish_reason'],'constraint':out.get('constraint')})
            if document_id:rows[-1]=store.save_generated(document_id,rows[-1])
            progress[request_id]['completed']=len(rows)
        return {'task':body.task,'original':body.text,'segments':rows,'text':assemble(body.text,rows),
                'document_id':document_id,'complete':all(r.get('finish_reason','stop')=='stop' or r['status']=='user_confirmed' for r in rows),
                'completed_segments':sum(r.get('finish_reason','stop')=='stop' or r['status']=='user_confirmed' for r in rows),'total_segments':len(parts),
                'review_required':sum(x['status']=='review_required' for x in rows),'model':model_name,'request_id':request_id}

    @app.post('/v1/chat/completions',dependencies=[Depends(auth)])
    async def chat(body:ChatRequest,request:Request):
        if body.model and body.model!=model_name:raise HTTPException(400,'未知模型。')
        if any(m.role not in {'system','user','assistant'} for m in body.messages):raise HTTPException(400,'消息角色无效。')
        turns=[m.model_dump() for m in body.messages];request_id='chatcmpl-'+uuid.uuid4().hex;created=int(time.time())
        if not body.stream:
            out=await await_result(turns,body.max_tokens,request)
            return {'id':request_id,'object':'chat.completion','created':created,'model':model_name,
                    'choices':[{'index':0,'message':{'role':'assistant','content':out['text']},'finish_reason':out['finish_reason']}],
                    'usage':{'prompt_tokens':out['prompt_tokens'],'completion_tokens':out['completion_tokens'],
                             'total_tokens':out['prompt_tokens']+out['completion_tokens']}}
        if hasattr(engine,'make_streamer'):streamer=engine.make_streamer()
        else:
            from transformers import TextIteratorStreamer
            streamer=TextIteratorStreamer(engine.tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=.2)
        cancel=threading.Event();task=await start(turns,body.max_tokens,cancel,streamer)
        def next_chunk():
            try:return 'chunk',next(streamer)
            except StopIteration:return 'end',''
            except queue.Empty:return 'wait',''
        def event(delta,finish=None,usage=None):
            data={'id':request_id,'object':'chat.completion.chunk','created':created,'model':model_name,
                  'choices':[] if usage else [{'index':0,'delta':delta,'finish_reason':finish}]}
            if usage:data['usage']=usage
            return 'data: '+json.dumps(data,ensure_ascii=False)+'\n\n'
        async def stream():
            try:
                yield event({'role':'assistant'})
                while True:
                    if await request.is_disconnected():cancel.set();break
                    kind,text=await asyncio.to_thread(next_chunk)
                    if kind=='chunk' and text:yield event({'content':text})
                    elif kind=='end' or kind=='wait' and task.done():break
                out=await asyncio.shield(task)
                yield event({},out['finish_reason'])
                if (body.stream_options or {}).get('include_usage'):
                    yield event({},usage={'prompt_tokens':out['prompt_tokens'],'completion_tokens':out['completion_tokens'],
                                          'total_tokens':out['prompt_tokens']+out['completion_tokens']})
                yield 'data: [DONE]\n\n'
            except asyncio.CancelledError:cancel.set();raise
            except Exception as e:
                yield 'data: '+json.dumps({'error':{'type':type(e).__name__,'message':'生成未完成，请重试。'}})+'\n\n'
                yield 'data: [DONE]\n\n'
            finally:
                if not task.done():cancel.set()
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-store'})
    return app


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='clean_2k_seed24');ap.add_argument('--port',type=int,default=8010)
    ap.add_argument('--reader-task-timeout',type=float,default=300)
    ap.add_argument('--backend',choices=['hf','native'],default='native');ap.add_argument('--native-port',type=int,default=11436)
    ap.add_argument('--unconstrained',action='store_true')
    args=ap.parse_args()
    if args.backend=='native':
        from classical.native import NativeEngine
        engine=NativeEngine(args.native_port,constrain=not args.unconstrained)
        if args.model!=engine.model_name:raise ValueError('请求模型与已加载 GGUF 不匹配。')
        native_state=json.loads((ROOT/'data/native-process.json').read_text(encoding='utf-8-sig'))
        engine.backend_description='llama.cpp '+native_state['variant']+'; reader punctuation '+('constrained' if engine.constrain else 'free')
    else:
        from classical.inference import HFEngine
        engine=HFEngine(1)
        if args.model!='base':engine.select(args.model,ROOT/'weights'/args.model)
    path=ROOT/'data/access.token'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_text(secrets.token_hex(32),encoding='utf-8')
    import uvicorn
    from classical.documents import DocumentStore
    uvicorn.run(create_app(engine,path.read_text().strip(),args.model,reader_task_timeout=args.reader_task_timeout,
                          store=DocumentStore(ROOT/'data/reader.sqlite')),host='127.0.0.1',port=args.port,workers=1)


if __name__=='__main__':main()
