"""Client TTFT and actual completion token usage; test the full local gateway."""
import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,write_json,write_jsonl


def request_one(url,token,model,prompt):
    import requests
    start=time.perf_counter();first=None;usage=None;text='';error=None;finish=None
    try:
        with requests.post(url,headers={'Authorization':'Bearer '+token},json={
            'model':model,'messages':[{'role':'user','content':prompt}], 'stream':True,'max_tokens':128,
            'temperature':0,'stream_options':{'include_usage':True}},stream=True,timeout=(5,150)) as response:
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1):
                if not raw.startswith(b'data: '):continue
                value=raw[6:].decode('utf-8')
                if value=='[DONE]':break
                data=json.loads(value)
                if data.get('error'):raise RuntimeError(data['error']['type'])
                if data.get('usage'):usage=data['usage']
                for choice in data.get('choices',[]):
                    content=choice.get('delta',{}).get('content','')
                    if content:
                        if first is None:first=time.perf_counter()
                        text+=content
                    if choice.get('finish_reason'):finish=choice['finish_reason']
        if not usage or first is None:raise RuntimeError('Missing actual token usage or content')
    except Exception as exc:error=type(exc).__name__
    end=time.perf_counter()
    return {'ok':error is None,'error':error,'ttft_ms':(first-start)*1000 if first else None,
        'total_ms':(end-start)*1000,'output_tokens':usage['completion_tokens'] if usage else None,
        'finish_reason':finish,'output_characters':len(text)}


def percentile(values,p):
    if not values:return None
    values=sorted(values);return values[min(len(values)-1,int((len(values)-1)*p))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8010);ap.add_argument('--model',default='clean_2k_seed24')
    ap.add_argument('--requests',type=int,default=30);ap.add_argument('--levels',default='1,2')
    ap.add_argument('--output-prefix',default='reports/gateway-benchmark');args=ap.parse_args()
    output=(ROOT/args.output_prefix).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Report must stay in this project')
    token=(ROOT/'data/access.token').read_text().strip();url=f'http://127.0.0.1:{args.port}/v1/chat/completions'
    import requests
    health=requests.get(f'http://127.0.0.1:{args.port}/health',timeout=10).json()
    if health['model']!=args.model:raise ValueError('Benchmark model does not match loaded service')
    prompts=['为下列古文断句，只输出原文：学而时习之不亦说乎有朋自远方来不亦乐乎',
             '把下面古文译成白话，只输出译文：人不知而不愠，不亦君子乎。',
             '把下面白话译成文言，只输出译文：学习了知识以后，还要经常复习。',
             '为下列古文断句，只输出原文：知之者不如好之者好之者不如乐之者']
    warmup=request_one(url,token,args.model,prompts[0]);assert warmup['ok'],warmup
    all_rows=[];levels=[];expected_levels=list(map(int,args.levels.split(',')))
    for n in expected_levels:
        started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            rows=list(pool.map(lambda i:request_one(url,token,args.model,prompts[i%len(prompts)]),range(args.requests)))
        duration=time.perf_counter()-started
        for i,row in enumerate(rows):row.update(concurrency=n,index=i)
        all_rows.extend(rows);good=[x for x in rows if x['ok']]
        result={'concurrency':n,'requests':len(rows),'successful':len(good),'errors':len(rows)-len(good),'wall_seconds':duration,
                'rps':len(good)/duration,'ttft_p50_ms':percentile([x['ttft_ms'] for x in good],.5),
                'ttft_p95_ms':percentile([x['ttft_ms'] for x in good],.95),'total_p95_ms':percentile([x['total_ms'] for x in good],.95),
                'actual_output_tokens':sum(x['output_tokens'] for x in good)}
        levels.append(result);print(result,flush=True)
        write_jsonl(Path(str(output)+'-samples.jsonl'),all_rows)
        write_json(Path(str(output)+'.json'),{'target':'full authenticated local gateway, not direct engine','model':args.model,'health':health,
             'complete':len(levels)==len(expected_levels),'requested_levels':expected_levels,'requests_per_level':args.requests,
             'levels':levels,'workload':'4 repeated short prompts, warmed model, output limit 128, low sample count; not production traffic or pure decoder throughput'})


if __name__=='__main__':main()
