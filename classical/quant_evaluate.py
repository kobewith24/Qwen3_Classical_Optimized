"""Same-checkpoint GGUF Q8/Q4 comparison on an isolated local Ollama server."""
import argparse
import json
import time
from collections import defaultdict

import requests
from .core import ROOT,digest,file_digest,messages,read_jsonl,score,write_json,write_jsonl
from .evaluate import paired_ci,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=11435)
    ap.add_argument('--models',default='classical-clean-q8,classical-clean-q4');ap.add_argument('--selection',default='primary')
    ap.add_argument('--limit-per-label',type=int,default=0);args=ap.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    version=requests.get(url+'/api/version',timeout=5).json()['version']
    selection=json.loads((ROOT/'reports'/f'eval-{args.selection}-selection.json').read_text(encoding='utf-8'))
    by_id={x['id']:x for x in read_jsonl(ROOT/'data/prepared/test.jsonl')};items=[];counts=defaultdict(int)
    hf_prompts={x['id']:x['prompt_tokens'] for x in read_jsonl(ROOT/'reports'/f'eval-{args.selection}-base-samples.jsonl')}
    for key in selection['ids']:
        item=by_id[key];group=(item['suite'],item.get('label',item['task']))
        if args.limit_per_label and counts[group]>=args.limit_per_label:continue
        counts[group]+=1;items.append(item)
    write_json(ROOT/'reports/quant-evaluation-selection.json',{'ids':[x['id'] for x in items],
        'primary_selection_sha256':file_digest(ROOT/'reports'/f'eval-{args.selection}-selection.json'),
        'limit_per_label':args.limit_per_label,'models':args.models})
    cache_path=ROOT/'cache/quant-answers.jsonl'
    cache={x['key']:x for x in read_jsonl(cache_path)} if cache_path.exists() else {}
    results={};all_rows={};template_hash=None
    tags=requests.get(url+'/api/tags',timeout=10).json()['models']
    digests={x['name']:x['digest'] for x in tags}
    for name in args.models.split(','):
        info=requests.post(url+'/api/show',json={'model':name},timeout=20).json()
        template=digest(info.get('template',''))
        if template_hash is not None and template!=template_hash:raise ValueError('Quantization models use different chat templates')
        template_hash=template
        model_digest=digests.get(name) or digests.get(name+':latest')
        if not model_digest:raise ValueError('Model digest not found')
        rows=[]
        for index,item in enumerate(items):
            options={'temperature':0,'seed':24,'num_ctx':2048,'num_predict':512 if item['task']=='punctuate' else 256}
            key=digest({'digest':model_digest,'ollama':version,'template':template,'messages':messages(item),'options':options,'think':False})
            if key not in cache:
                started=time.perf_counter()
                response=requests.post(url+'/api/chat',json={'model':name,'messages':messages(item),'stream':False,'think':False,
                                      'options':options,'keep_alive':'5m'},timeout=(5,180),allow_redirects=False)
                response.raise_for_status();data=response.json()
                if not data.get('done') or not data.get('eval_count'):raise ValueError('Missing completed generation or actual token count')
                answer={'key':key,'text':data['message']['content'].strip(),'model_digest':model_digest,
                        'completion_tokens':data['eval_count'],'prompt_tokens':data.get('prompt_eval_count'),
                        'eval_duration_ns':data.get('eval_duration'),'prompt_eval_duration_ns':data.get('prompt_eval_duration'),
                        'load_duration_ns':data.get('load_duration'),'wall_seconds':time.perf_counter()-started,
                        'finish_reason':data.get('done_reason','unknown')}
                with cache_path.open('a',encoding='utf-8') as f:f.write(json.dumps(answer,ensure_ascii=False)+'\n');f.flush()
                cache[key]=answer
            answer=cache[key]
            if answer['prompt_tokens']!=hf_prompts[item['id']]:
                raise ValueError(f"Native prompt token count differs from the frozen HF template for {item['id']}: {answer['prompt_tokens']} != {hf_prompts[item['id']]}; inspect serialization before comparing quality")
            rows.append({**item,**answer,'expected_prompt_tokens':hf_prompts[item['id']],
                         'metrics':score(item,answer['text'])})
            write_jsonl(ROOT/'reports'/f'quant-{name}-samples.jsonl',rows)
            print(name,index+1,'/',len(items),flush=True)
        total_tokens=sum(x['completion_tokens'] for x in rows);duration=sum(x['eval_duration_ns'] for x in rows)/1e9
        results[name]={'groups':summary(rows),'actual_generated_tokens':total_tokens,
                       'prompt_token_counts_match_hf':True,
                       'decoder_tokens_per_second':total_tokens/duration,'decoder_seconds':duration,
                       'model_digest':model_digest,'template_hash':template}
        all_rows[name]=rows
        write_json(ROOT/'reports/quant-comparison.json',{'models':results,'ollama_version':version,'context':2048,
                   'warning':'Same merged checkpoint and backend; quality reported per task, no heterogeneous mean. Includes warmed and initial requests.'})
        ps=requests.get(url+'/api/ps',timeout=5).json();write_json(ROOT/'reports'/f'ollama-residency-{name}.json',ps)
    names=list(all_rows);paired={}
    pairs=[(names[0],name) for name in names[1:]]
    if len(names)>=3:pairs.append((names[-2],names[-1]))
    for first,second in pairs:
        comparison={}
        for group in results[first]['groups']:
            group_key=lambda x:f"{x['suite']}/{x.get('label',x['task'])}"
            before={x['id']:x['metrics']['score'] for x in all_rows[first] if group_key(x)==group}
            after={x['id']:x['metrics']['score'] for x in all_rows[second] if group_key(x)==group}
            comparison[group]=paired_ci(before,after)
        paired[second+' minus '+first]=comparison
    write_json(ROOT/'reports/quant-comparison.json',{'models':results,'paired_comparisons':paired,
        'ollama_version':version,'context':2048,'sample_count_per_model':len(items),
        'metric_code_sha256':file_digest(ROOT/'classical/core.py'),
        'warning':'Actual eval_count/eval_duration used; per-task scores only; no claim of 1% aggregate quality loss.'})
    print('QUANT_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
