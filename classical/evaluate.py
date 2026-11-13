"""Paired, fingerprinted generation with sample-level scores and uncertainty."""
import argparse
import json
import random
import re
import time
from collections import defaultdict

from .core import ROOT,digest,file_digest,messages,read_jsonl,score,write_json,write_jsonl
from .prepare import OverlapIndex


def task_names(split_kind,requested=None):
    if requested is not None:return set(requested.split(','))
    return {'c2m','m2c','punctuate'} if split_kind=='dev' else {'translate','punctuate'}


class AnswerCache:
    def __init__(self,path):
        self.path=path; self.rows={x['key']:x for x in read_jsonl(path)} if path.exists() else {}

    @staticmethod
    def key(model_fingerprint,item,generation):
        return digest({'model':model_fingerprint,'messages':messages(item),'sample':item['id'],
                       'reference':item['reference'],'generation':generation})

    def put(self,row):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        self.rows[row['key']]=row


def paired_ci(before,after,seed=24,repeats=1000):
    common=sorted(set(before)&set(after)); diffs=[after[k]-before[k] for k in common]
    if not diffs:return None
    rng=random.Random(seed); means=sorted(sum(rng.choices(diffs,k=len(diffs)))/len(diffs) for _ in range(repeats))
    return {'n':len(diffs),'mean_delta':sum(diffs)/len(diffs),'bootstrap_95_ci':[means[int(.025*repeats)],means[int(.975*repeats)]],
            'unit':'sample within this frozen set; does not measure seed-to-seed training variance'}


def summary(rows):
    groups=defaultdict(list)
    for row in rows:groups[f"{row['suite']}/{row.get('label',row['task'])}"].append(row)
    result={}
    for group,items in groups.items():
        good=[x for x in items if not x.get('error')]
        s={'n':len(items),'successful':len(good),'errors':len(items)-len(good)}
        if good:
            s['score']=sum(x['metrics']['score'] for x in good)/len(good)
            if items[0]['task']=='punctuate':
                s['body_preservation_rate']=sum(x['metrics']['body_preserved'] for x in good)/len(good)
                s['mean_body_character_error_rate']=sum(x['metrics']['body_character_error_rate'] for x in good)/len(good)
                s['boundary_f1']=sum(x['metrics']['boundary_f1'] for x in good)/len(good)
                s['typed_f1']=sum(x['metrics']['typed_f1'] for x in good)/len(good)
            s['length_truncated']=sum(x.get('finish_reason')=='length' for x in good)
        result[group]=s
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',default='base,clean_2k_seed24')
    ap.add_argument('--limit-per-label',type=int,default=40);ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--tasks',default=None);ap.add_argument('--tag',default='primary')
    ap.add_argument('--reuse-selection',action='store_true',help='Reuse exact frozen IDs and verify the test file hash; no resampling')
    ap.add_argument('--data-file',default='data/prepared/test.jsonl')
    ap.add_argument('--split-kind',choices=['test','dev'],default='test')
    args=ap.parse_args()
    selected_tasks=task_names(args.split_kind,args.tasks);args.tasks=','.join(sorted(selected_tasks))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):raise ValueError('Invalid evaluation tag')
    data_file=(ROOT/args.data_file).resolve()
    if not data_file.is_relative_to(ROOT.resolve()):raise ValueError('Evaluation input must stay in this project')
    items=list(read_jsonl(data_file));kept=[];groups=defaultdict(list)
    if args.split_kind=='dev':
        if 'legacy' in args.models.split(','):raise ValueError('Legacy exposure is not controlled on this development split')
        items=[{**x,'suite':'dev','label':x['task']} for x in items]
    legacy=list(read_jsonl(ROOT/'assets/source_dataset/train.jsonl'))[:2000]
    exposure=OverlapIndex(t for x in legacy for t in [x['messages'][1]['content'].split('\n',1)[-1],x['messages'][-1]['content']])
    excluded=[]
    for item in items:
        if item['task'] not in selected_tasks:continue
        if args.split_kind=='test' and (exposure.matches(item['input']) or exposure.matches(item['reference'])):
            excluded.append(item['id']);continue
        group=(item['suite'],item.get('label',item['task']))
        groups[group].append(item)
    sampling=random.Random(20261112)
    batches=[]
    for group,rows in sorted(groups.items()):
        sampling.shuffle(rows)
        chosen=rows[:args.limit_per_label] if args.limit_per_label else rows
        kept.extend(chosen)
        batches.extend(chosen[start:start+args.batch_size] for start in range(0,len(chosen),args.batch_size))
    selection_path=ROOT/'reports'/f'eval-{args.tag}-selection.json'
    if args.reuse_selection:
        selection=json.loads(selection_path.read_text(encoding='utf-8'))
        if selection.get('data_sha256',selection.get('test_sha256'))!=file_digest(data_file):
            raise ValueError('Frozen test data changed; do not reuse the selection')
        by_id={x['id']:x for x in items};ids=selection['ids']
        if not ids or len(set(ids))!=len(ids):raise ValueError('Invalid frozen IDs')
        kept=[by_id[key] for key in ids]
        if args.split_kind=='test' and any(exposure.matches(x['input']) or exposure.matches(x['reference']) for x in kept):
            raise ValueError('Frozen selection conflicts with the legacy exposure check; audit before rerunning')
        frozen_groups=defaultdict(list);batches=[]
        for item in kept:frozen_groups[(item['suite'],item.get('label',item['task']))].append(item)
        for rows in frozen_groups.values():
            batches.extend(rows[start:start+args.batch_size] for start in range(0,len(rows),args.batch_size))
        print('REUSING_FROZEN_SELECTION',len(kept),flush=True)
    else:
        # Freeze exact evaluation IDs; changing a limit cannot silently reuse positional cache.
        write_json(selection_path,{'ids':[x['id'] for x in kept],
            'sampling_seed':20261112,'limit_per_label':args.limit_per_label,'tasks':args.tasks,
            'excluded_for_legacy_exposure':excluded,'legacy_exposure_policy':'first 2000 rows of the saved source dataset; old run lacks a dataset hash',
            'data_sha256':file_digest(data_file),'data_file':str(data_file.relative_to(ROOT)),'split_kind':args.split_kind})
    if not kept:raise ValueError('No samples selected; check task names and split kind')
    from .inference import HFEngine
    engine=HFEngine(args.batch_size);cache=AnswerCache(ROOT/'cache/answers.jsonl');results={};all_rows={}
    for name in args.models.split(','):
        adapter=ROOT/'assets/reference_adapter' if name=='legacy' else ROOT/'weights'/name
        if name!='base' and not (adapter/'adapter_model.safetensors').exists():raise FileNotFoundError(adapter)
        engine.select(name,adapter);fingerprint=engine.fingerprint();rows=[]
        for batch in batches:
            generation={'greedy':True,'max_new_tokens':512 if batch[0]['task']=='punctuate' else 256}
            missing=[]
            for item in batch:
                key=cache.key(fingerprint,item,generation)
                if key not in cache.rows:missing.append((key,item))
            if missing:
                out=engine.generate_items([x[1] for x in missing],generation['max_new_tokens'])
                for (key,item),answer in zip(missing,out):cache.put({'key':key,'model_fingerprint':fingerprint,'id':item['id'],**answer})
            for item in batch:
                key=cache.key(fingerprint,item,generation);answer=cache.rows[key]
                rows.append({**item,**answer,'metrics':score(item,answer['text'])})
            write_jsonl(ROOT/'reports'/f'eval-{args.tag}-{name}-samples.jsonl',rows)
            print(name,len(rows),'/',len(kept),flush=True)
        results[name]=summary(rows);all_rows[name]=rows
        write_json(ROOT/'reports'/f'eval-{args.tag}.json',{'protocol':'same NF4 base, greedy, explicit common messages, exact sample IDs; metrics v2 strict body',
            'models':results,'selection':str(ROOT/'reports'/f'eval-{args.tag}-selection.json')})
    comparisons={}
    if 'base' in all_rows:
        for name,rows in all_rows.items():
            if name=='base':continue
            comparisons[name]={}
            for group in results[name]:
                group_key=lambda x:f"{x['suite']}/{x.get('label',x['task'])}"
                before={x['id']:x['metrics']['score'] for x in all_rows['base'] if group_key(x)==group}
                after={x['id']:x['metrics']['score'] for x in rows if group_key(x)==group}
                comparisons[name][group]=paired_ci(before,after)
    report={'models':results,'paired_vs_base':comparisons,
        'split_kind':args.split_kind,'data_sha256':file_digest(data_file),
        'evaluation_role':'development diagnostic, not an unseen test' if args.split_kind=='dev' else 'fixed paired evaluation',
        'metric_code_sha256':file_digest(ROOT/'classical/core.py'),'engine_code_sha256':file_digest(ROOT/'classical/inference.py'),
        'selection_sha256':file_digest(ROOT/'reports'/f'eval-{args.tag}-selection.json'),
        'warning':'No heterogeneous task average, no claim of saturation or general ability retention. Sample bootstrap does not cover unseen-book or training-seed variability.'}
    write_json(ROOT/'reports'/f'eval-{args.tag}.json',report)
    lines=['# Frozen-set paired evaluation','','NF4 baseline and adapters share prompts, templates and generation limits. Punctuation score is zero when the original body changes.','',
           '|Model|Set|n|Score|Body preserved|','|---|---|---:|---:|---:|']
    for name,groups in results.items():
        for group,s in groups.items():lines.append(f"|{name}|{group}|{s['n']}|{s.get('score',0):.4f}|{s.get('body_preservation_rate','—')}|")
    lines+=['','Paired bootstrap intervals and sample-level outputs are in the adjacent JSON files. These intervals do not quantify training seed variability.']
    (ROOT/'reports'/f'eval-{args.tag}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
