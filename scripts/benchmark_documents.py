"""Three natural-text lengths through authenticated reader; training text, performance only."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,digest,file_digest,read_jsonl,write_json
from classical.constraints import compile_punctuation


def main():
    training=ROOT/'data/prepared/train.jsonl';eligible=[];excluded=[]
    for row in read_jsonl(training):
        if row['task']!='punctuate':continue
        try:compile_punctuation(row['input'])
        except ValueError as e:excluded.append({'id':row['id'],'reason':str(e)});continue
        eligible.append(row)
    corpus='\n'.join(r['input'] for r in eligible)
    s=requests.Session();s.trust_env=False;url='http://127.0.0.1:8010';headers={'Authorization':'Bearer '+(ROOT/'data/access.token').read_text().strip()}
    health=s.get(url+'/health',timeout=10).json();results=[]
    for length in (500,1500,3000):
        text=corpus[:length];start=time.perf_counter()
        response=s.post(url+'/api/process',headers=headers,json={'task':'punctuate','text':text},timeout=330)
        elapsed=time.perf_counter()-start;response.raise_for_status();data=response.json()
        if any(r.get('cache_hit') for r in data['segments']):raise ValueError('Cold document result cache required; choose a new workload before rerunning')
        result={'characters':length,'seconds':elapsed,'segments':len(data['segments']),'complete':data['complete'],
                'body_preserved':body(text)==body(data['text']),'all_constraints_completed':all(r.get('constraint',{}).get('complete') for r in data['segments']),
                'new_output_tokens':sum(r['usage']['completion_tokens'] for r in data['segments']),
                'source_sha256':digest(text.encode()),'document_id':data['document_id']}
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        write_json(ROOT/'reports/validation/document-benchmark.json',{'health':health,'complete':len(results)==3,'results':results,
            'source_manifest':{'training_sha256':file_digest(training),'eligible_parent_ids':[r['id'] for r in eligible],'excluded':excluded},
            'scope':'one document per length; natural classical training excerpts, performance/contract only, no semantic quality score or latency percentile claim'})
    assert all(r['complete'] and r['body_preserved'] and r['all_constraints_completed'] for r in results)

if __name__=='__main__':main()
