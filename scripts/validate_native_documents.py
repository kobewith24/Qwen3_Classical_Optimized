"""Real durable-resume check across a deliberate gateway restart; no fixtures."""
import argparse
import concurrent.futures
import json
import sys
import time
import uuid
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['before','after']);args=ap.parse_args()
    s=requests.Session();s.trust_env=False;url='http://127.0.0.1:8010'
    h={'Authorization':'Bearer '+(ROOT/'data/access.token').read_text().strip()};folder=ROOT/'reports/validation'
    checks=[]
    def check(name,passed,detail=None):checks.append({'name':name,'passed':bool(passed),'detail':detail})
    if args.phase=='before':
        text='君子曰学不可以已青取之于蓝而青于蓝冰水为之而寒于水木直中绳輮以为轮其曲中规虽有槁暴不复挺者輮使之然也故木受绳则直金就砺则利君子博学而日参省乎己则知明而行无过矣吾尝终日而思矣不如须臾之所学也吾尝跂而望矣不如登高之博见也登高而招臂非加长也而见者远顺风而呼声非加疾也而闻者彰\n'+'人之为学不日进则日退独学无友则孤陋而难成久处一方则习染而不自觉矣'*18
        text+='\n【续跑验收'+uuid.uuid4().hex[:8]+'】'
        key=s.post(url+'/api/documents',headers=h,json={'task':'punctuate','text':text},timeout=10).json()['document_id']
        request_id=uuid.uuid4().hex
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(requests.post,url+'/api/process',headers=h,json={'task':'punctuate','text':text,'request_id':request_id},timeout=150)
            try:
                started=time.monotonic();observed=None
                while time.monotonic()-started<60:
                    response=s.get(url+'/api/progress/'+request_id,headers=h,timeout=10)
                    if response.status_code==200:
                        observed=response.json()
                        if observed['completed_segments']>=1:break
                    time.sleep(.1)
            finally:s.post(url+'/api/cancel',headers=h,json={'request_id':request_id},timeout=10).raise_for_status()
            result=future.result(timeout=20)
        data=s.get(url+'/api/documents/'+key,headers=h,timeout=10).json()
        check('cancelled response is explicit',result.status_code==409,result.status_code)
        check('completed prefix remains readable',0<data['completed_segments']<data['total_segments'],{'saved':data['completed_segments'],'total':data['total_segments']})
        row=data['segments'][0];edited=row['original']+'。'
        patch=s.patch(url+f'/api/documents/{key}/segments/0',headers=h,json={'text':edited,'revision':row['revision']},timeout=10)
        check('manual edit persisted',patch.status_code==200,patch.status_code)
        write_json(folder/'resume-before.json',{'document_id':key,'source':text,'edited':edited,'revision':patch.json()['revision'],'checks':checks})
    else:
        before=json.loads((folder/'resume-before.json').read_text(encoding='utf-8'));key=before['document_id'];text=before['source']
        restored=s.get(url+'/api/documents/'+key,headers=h,timeout=10);restored.raise_for_status();data=restored.json()
        check('revision and manual edit survive process restart',data['segments'][0]['text']==before['edited'] and data['segments'][0]['revision']==before['revision'])
        saved=data['completed_segments'];m0=s.get(url+'/metrics',headers=h,timeout=10).json()
        result=s.post(url+'/api/process',headers=h,json={'task':'punctuate','text':text},timeout=300);result.raise_for_status();finished=result.json()
        m1=s.get(url+'/metrics',headers=h,timeout=10).json()
        check('resumption only generates missing segments',m1['completed']-m0['completed']==data['total_segments']-saved,{'saved_before':saved,'new_generations':m1['completed']-m0['completed']})
        check('complete resumed text keeps source body',finished['complete'] and body(finished['text'])==body(text))
        check('cached prefix has zero new token usage',all(r['usage']['completion_tokens']==0 for r in finished['segments'] if r.get('cache_hit')))
        stale=s.patch(url+f'/api/documents/{key}/segments/0',headers=h,json={'text':before['edited'],'revision':0},timeout=10)
        check('stale edit rejected after restart',stale.status_code==409)
        again=s.post(url+'/api/process',headers=h,json={'task':'punctuate','text':text},timeout=30);again.raise_for_status();m2=s.get(url+'/metrics',headers=h,timeout=10).json()
        check('fully cached document generates no new tokens',m2['completed']==m1['completed'] and all(r['usage']['completion_tokens']==0 for r in again.json()['segments']))
        write_json(folder/'resume-after.json',{'document_id':key,'checks':checks,'metrics':m2})
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    assert checks and all(r['passed'] for r in checks)

if __name__=='__main__':main()
