"""Same GGUF, prompts and limits: source grammar versus free decoding on fixed dev IDs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,INSTRUCTIONS,body,digest,file_digest,read_jsonl,score,write_json,write_jsonl
from classical.evaluate import paired_ci,summary
from classical.native import NativeEngine


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=11436)
    parser.add_argument('--tag',required=True);parser.add_argument('--fewshot',action='store_true')
    parser.add_argument('--holdout',action='store_true');args=parser.parse_args()
    if not args.tag.replace('-','').replace('_','').isalnum():raise ValueError('Invalid report tag')
    folder=ROOT/'reports/validation';data=ROOT/'data/dev_review_20261113/items.jsonl'
    items=[{**r,'suite':'dev','label':'punctuate'} for r in read_jsonl(data) if r['task']=='punctuate']
    group_key='dev/punctuate'
    if args.holdout:
        if args.fewshot:raise ValueError('Holdout protocol fixes the zero-shot prompt')
        data=ROOT/'data/prepared/test.jsonl';frozen=json.loads((ROOT/'reports/validation/holdout-selection.json').read_text(encoding='utf-8'))
        if file_digest(data)!=frozen['data_sha256']:raise ValueError('Frozen holdout changed')
        available={r['id']:r for r in read_jsonl(data)}
        items=[{**available[key],'origin_suite':available[key]['suite'],'suite':'holdout','label':'punctuate'} for key in frozen['ids']]
        group_key='holdout/punctuate'
    selection={'ids':[r['id'] for r in items],'data_sha256':file_digest(data),'role':'development diagnostic, not unseen test',
               'predeclared_gate':{'strict_f1_delta_min':-.01,'constrained_completion_rate_min':.98,'constrained_body_preservation_min':.98},
               'generation':{'greedy':True,'seed':24,'cache_prompt':False,'max_new_tokens':512},
               'source_sha256':{p:file_digest(ROOT/p) for p in ['classical/native.py','classical/constraints.py','classical/core.py']}}
    if args.holdout:
        selection['role']=frozen['role'];selection['holdout_manifest_sha256']=file_digest(ROOT/'reports/validation/holdout-selection.json')
    if args.fewshot:
        training=ROOT/'data/prepared/train.jsonl';examples=[];books=set()
        for row in sorted((r for r in read_jsonl(training) if r['task']=='punctuate'),key=lambda r:(len(r['input']),r['id'])):
            if row['book'] in books or body(row['input'])!=body(row['reference']):continue
            examples.append(row);books.add(row['book'])
            if len(examples)==2:break
        prefix=INSTRUCTIONS['punctuate']+'\n以下示例只演示格式；保持原字及顺序，仅插入标点。\n'
        prefix+='\n'.join('示例原文：'+r['input']+'\n示例断句：'+r['reference'] for r in examples)+'\n只处理下面的实际原文：'
        items=[{**r,'instruction':prefix} for r in items]
        selection['fewshot']={'ids':[r['id'] for r in examples],'training_sha256':file_digest(training),'instruction':prefix}
    selection_path=folder/(args.tag+'-selection.json')
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding='utf-8'))!=selection:raise ValueError('Frozen native protocol changed; use another tag')
    else:write_json(selection_path,selection)
    engine=NativeEngine(args.port);results={};all_rows={};fingerprints={}
    cache_path=ROOT/'cache/native-answers.jsonl';cache={r['key']:r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    for mode in ('free','constrained'):
        engine.constrain=mode=='constrained';rows=[];fingerprint=engine.fingerprint();fingerprints[mode]=fingerprint
        for item in items:
            key=digest({'model':fingerprint,'item':item,'protocol':selection})
            if key not in cache:
                try:
                    out=engine.generate_reader(item,max_new_tokens=512)
                    answer={**out,'error':None}
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    answer={'text':'','error':type(exc).__name__+': '+str(exc),'finish_reason':'error'}
                row={'key':key,**answer};cache_path.parent.mkdir(parents=True,exist_ok=True)
                with cache_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                cache[key]=row
            row={**item,**cache[key]};row['metrics']=score(item,row['text'])
            rows.append(row);write_jsonl(folder/(args.tag+'-'+mode+'-samples.jsonl'),rows)
            print(engine.model_name,mode,len(rows),'/',len(items),flush=True)
        stats=summary(rows)[group_key]
        # Include failures in the strict task score; no survivor-only average.
        stats['all_sample_strict_f1']=sum(r['metrics']['score'] if not r['error'] else 0 for r in rows)/len(rows)
        stats['completed_rate']=sum(not r['error'] and r['finish_reason']=='stop' for r in rows)/len(rows)
        stats['all_sample_body_preservation']=sum(not r['error'] and r['finish_reason']=='stop' and r['metrics']['body_preserved'] for r in rows)/len(rows)
        results[mode]=stats;all_rows[mode]=rows
    delta=results['constrained']['all_sample_strict_f1']-results['free']['all_sample_strict_f1']
    gate=delta>=-.01 and results['constrained']['completed_rate']>=.98 and results['constrained']['all_sample_body_preservation']>=.98
    report={'model':engine.model_name,'fingerprints':fingerprints,'models':results,'complete':True,'development_gate_passed':gate,
            'paired_strict_f1':paired_ci({r['id']:r['metrics']['score'] for r in all_rows['free']},{r['id']:r['metrics']['score'] for r in all_rows['constrained']}),
            'protocol_sha256':file_digest(selection_path),'warning':'Development-set diagnostic. Structural body invariance is a decoding contract, not learned semantic accuracy or an unseen-test claim.'}
    if args.holdout:
        grouped={}
        for origin in ('bench','ood'):
            per_mode={mode:[r for r in rows if r['origin_suite']==origin] for mode,rows in all_rows.items()}
            means={mode:sum(r['metrics']['score'] if not r['error'] else 0 for r in rows)/len(rows) for mode,rows in per_mode.items()}
            completed=sum(not r['error'] and r['finish_reason']=='stop' and r['metrics']['body_preserved'] for r in per_mode['constrained'])/len(per_mode['constrained'])
            grouped[origin]={'n':len(per_mode['free']),'strict_f1':means,'delta':means['constrained']-means['free'],'completed_body_rate':completed,
                             'passed':means['constrained']-means['free']>=-.01 and completed>=.98}
        report['incremental_holdout']=grouped;report['holdout_gate_passed']=all(r['passed'] for r in grouped.values())
        report['warning']=frozen['role']+'. References structurally checked; no independent expert punctuation certification.'
    write_json(folder/(args.tag+'.json'),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
