"""Real model/gateway checks; no mocks, no writes outside this project."""
import argparse
import concurrent.futures
import datetime
import http.client
import json
import sys
import time
import uuid
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,file_digest,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8010)
    ap.add_argument('--output',default='reports/live-api-validation.json')
    ap.add_argument('--deadline-only',action='store_true');args=ap.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Report must stay in this project')
    url=f'http://127.0.0.1:{args.port}';token=(ROOT/'data/access.token').read_text().strip()
    headers={'Authorization':'Bearer '+token};checks=[];client=requests.Session();client.trust_env=False
    def check(name,passed,detail=None):checks.append({'name':name,'passed':bool(passed),'detail':detail})
    health=client.get(url+'/health',timeout=10);health.raise_for_status();health=health.json()
    def save():
        native_state=json.loads((ROOT/'data/native-process.json').read_text(encoding='utf-8-sig')) if health.get('backend','').startswith('llama.cpp') else None
        write_json(output,{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':health['model'],
            'health':health,'checks':checks,'mode':'deadline-only' if args.deadline_only else 'normal',
            'native_identity':{k:native_state[k] for k in ['variant','model_sha256','binary_sha256']} if native_state else None,
            'adapter_sha256':file_digest(ROOT/'weights'/health['model']/'adapter_model.safetensors') if health['model']!='base' else None,
            'code_sha256':{name:file_digest(ROOT/name) for name in ['api/server.py','classical/inference.py','scripts/validate_live_api.py']}})
        print(json.dumps(checks,ensure_ascii=False,indent=2))
        assert all(x['passed'] for x in checks),'Inspect real API checks before declaring completion'
    long_source='古之学者必有师师者所以传道受业解惑也人非生而知之者孰能无惑惑而不从师其为惑也终不解矣'*14
    if args.deadline_only:
        if not 0<health['reader_task_timeout_seconds']<=5:raise ValueError('Start a dedicated short-deadline server first')
        key=uuid.uuid4().hex;before=client.get(url+'/metrics',headers=headers,timeout=10).json()
        started=time.perf_counter()
        response=requests.post(url+'/api/process',headers=headers,json={'task':'punctuate','text':long_source,'request_id':key},timeout=20)
        elapsed=time.perf_counter()-started
        while time.perf_counter()-started<20:
            after=client.get(url+'/metrics',headers=headers,timeout=10).json()
            if after['active']==0 and after['reader_jobs']==0:break
            time.sleep(.05)
        check('real whole-document deadline returns explicit 504',response.status_code==504 and '整篇' in response.json().get('detail',''),
              {'status':response.status_code,'response_seconds':elapsed,'budget_seconds':health['reader_task_timeout_seconds']})
        check('real timed-out worker stops and releases capacity',after['active']==0 and after['reader_jobs']==0
              and after['reader_timeouts']==before['reader_timeouts']+1 and after['cancelled']>before['cancelled'],
              {'seconds_to_idle':time.perf_counter()-started,'metrics':after})
        check('timed-out progress removed',client.get(url+'/api/progress/'+key,headers=headers,timeout=10).status_code==404)
        save();return
    check('chat requires authentication',requests.post(url+'/v1/chat/completions',json={'messages':[{'role':'user','content':'断句'}]},timeout=10).status_code==401)
    check('wrong token rejected',client.get(url+'/v1/models',headers={'Authorization':'Bearer wrong'},timeout=10).status_code==401)
    models=client.get(url+'/v1/models',headers=headers,timeout=10);models.raise_for_status();model=models.json()['data'][0]['id']
    source='学而时习之不亦说乎有朋自远方来不亦乐乎'
    response=requests.post(url+'/api/process',headers=headers,json={'task':'punctuate','text':source},timeout=150);response.raise_for_status();data=response.json()
    check('real reader response retains original body',body(data['text'])==body(source),{'segments':len(data['segments']),'review_required':data['review_required']})
    check('reader usage distinguishes cached and generated output',all(
        (x.get('cache_hit') and x['usage']['completion_tokens']==0 and x['generation_usage']['completion_tokens']>0)
        or (not x.get('cache_hit') and x['usage']['completion_tokens']>0) for x in data['segments']))
    chunks=[];done=False;usage=None
    with requests.post(url+'/v1/chat/completions',headers=headers,json={'model':model,'messages':[{'role':'user','content':'把“学而时习之，不亦说乎”译成白话，只输出译文。'}],
        'stream':True,'max_tokens':128,'stream_options':{'include_usage':True}},stream=True,timeout=(5,150)) as stream:
        stream.raise_for_status()
        for raw in stream.iter_lines(chunk_size=1):
            if not raw.startswith(b'data: '):continue
            value=raw[6:].decode('utf-8')
            if value=='[DONE]':done=True;break
            item=json.loads(value)
            if item.get('usage'):usage=item['usage']
            for choice in item.get('choices',[]):
                if choice.get('delta',{}).get('content'):chunks.append(choice['delta']['content'])
    check('real SSE content and DONE',done and bool(''.join(chunks)))
    check('real SSE reports token IDs, not chunks',usage and usage['completion_tokens']>0,{'usage':usage,'content_chunks':len(chunks)})
    before=client.get(url+'/metrics',headers=headers,timeout=10).json()
    stream=requests.post(url+'/v1/chat/completions',headers=headers,json={'model':model,'messages':[{'role':'user','content':'只给下列原文添加标点，不解释，不省略：\n'+long_source}],
        'stream':True,'max_tokens':640},stream=True,timeout=(5,150))
    stream.raise_for_status();first=False
    for raw in stream.iter_lines(chunk_size=1):
        if raw.startswith(b'data: ') and raw[6:]!=b'[DONE]':
            item=json.loads(raw[6:])
            if any(c.get('delta',{}).get('content') for c in item.get('choices',[])):first=True;break
    active_before=client.get(url+'/metrics',headers=headers,timeout=10).json()['active']
    started=time.perf_counter();stream.close();after=before
    while time.perf_counter()-started<15:
        after=client.get(url+'/metrics',headers=headers,timeout=10).json()
        if after['active']==0:break
        time.sleep(.1)
    check('disconnect stops real worker and recovers capacity',first and active_before==1 and after['active']==0 and after['cancelled']>before['cancelled'],
          {'active_before_disconnect':active_before,'seconds_to_idle':time.perf_counter()-started,'cancelled_before':before['cancelled'],'cancelled_after':after['cancelled'],
           'scope':'worker completion and stop signal; not an independent GPU kernel stop measurement'})
    # Explicit cancellation must work while the original JSON request remains connected.
    # Use a finite, nonrepetitive first paragraph to observe a completed segment;
    # the repeated stress input above may keep a segment generating for >60 s.
    first_paragraph='君子曰学不可以已青取之于蓝而青于蓝冰水为之而寒于水木直中绳輮以为轮其曲中规虽有槁暴不复挺者輮使之然也故木受绳则直金就砺则利君子博学而日参省乎己则知明而行无过矣吾尝终日而思矣不如须臾之所学也吾尝跂而望矣不如登高之博见也登高而招臂非加长也而见者远顺风而呼声非加疾也而闻者彰'
    progress_source=first_paragraph+'\n'+long_source
    key=uuid.uuid4().hex;before=client.get(url+'/metrics',headers=headers,timeout=10).json()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(requests.post,url+'/api/process',headers=headers,
            json={'task':'punctuate','text':progress_source,'request_id':key},timeout=150)
        started=time.perf_counter();active=0;observed=None
        try:
            while time.perf_counter()-started<60:
                active=client.get(url+'/metrics',headers=headers,timeout=10).json()['active']
                snapshot=client.get(url+'/api/progress/'+key,headers=headers,timeout=10)
                if snapshot.status_code==200:
                    observed=snapshot.json()
                    if active==1 and observed['completed_segments']>=1:break
                time.sleep(.5)
            check('progress endpoint requires authentication',client.get(url+'/api/progress/'+key,timeout=10).status_code==401)
            check('real progress counts completed segments without exposing text',observed and 0<observed['completed_segments']<observed['total_segments']
                  and set(observed)=={'request_id','completed_segments','total_segments','elapsed_seconds','task_timeout_seconds'},observed)
        except requests.RequestException as exc:check('progress observation transport',False,{'error':str(exc)})
        finally:cancel_response=client.post(url+'/api/cancel',headers=headers,json={'request_id':key},timeout=10)
        stopped=time.perf_counter();result=future.result(timeout=20)
        while time.perf_counter()-stopped<15:
            after=client.get(url+'/metrics',headers=headers,timeout=10).json()
            if after['active']==0 and after['reader_jobs']==0:break
            time.sleep(.05)
    check('explicit reader cancellation stops real connected request',active==1 and cancel_response.status_code==200
        and result.status_code==409 and after['active']==0 and after['reader_jobs']==0 and after['cancelled']>before['cancelled'],
        {'response_status':result.status_code,'seconds_to_idle':time.perf_counter()-stopped,
         'cancelled_before':before['cancelled'],'cancelled_after':after['cancelled']})
    check('finished progress removed',client.get(url+'/api/progress/'+key,headers=headers,timeout=10).status_code==404)
    # A direct non-stream client disconnect must also release the real worker.
    before=client.get(url+'/metrics',headers=headers,timeout=10).json()
    connection=http.client.HTTPConnection('127.0.0.1',args.port,timeout=10)
    connection.request('POST','/api/process',body=json.dumps({'task':'punctuate','text':long_source,'request_id':uuid.uuid4().hex}).encode(),
        headers={**headers,'Content-Type':'application/json'})
    started=time.perf_counter();active=0
    while time.perf_counter()-started<15:
        active=client.get(url+'/metrics',headers=headers,timeout=10).json()['active']
        if active==1:break
        time.sleep(.05)
    stopped=time.perf_counter();connection.close()
    while time.perf_counter()-stopped<15:
        after=client.get(url+'/metrics',headers=headers,timeout=10).json()
        if after['active']==0 and after['reader_jobs']==0:break
        time.sleep(.05)
    check('direct JSON disconnect stops real reader worker',active==1 and after['active']==0 and after['reader_jobs']==0
        and after['cancelled']>before['cancelled'],{'seconds_to_idle':time.perf_counter()-stopped})
    next_response=requests.post(url+'/api/process',headers=headers,json={'task':'c2m','text':'知之者不如好之者。'},timeout=150)
    check('reader capacity usable after cancellation',next_response.status_code==200)
    save()


if __name__=='__main__':main()
