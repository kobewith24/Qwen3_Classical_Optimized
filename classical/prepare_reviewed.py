"""Versioned reviewed recipe: preserve old data, fixed filtered dev and more copy windows."""
import argparse
import json
import random
from collections import Counter,defaultdict
from .core import ROOT,INSTRUCTIONS,body,digest,file_digest,read_jsonl,tokenize_record,write_json,write_jsonl
from .prepare import OverlapIndex,evaluation_items
from .quality import pair_key,corruption_flags,translation_flags,punctuation_windows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/reviewed');ap.add_argument('--samples',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=24);args=ap.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Output outside project')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model',local_files_only=True)
    review_path=ROOT/'configs/review_20261113.json';review=json.loads(review_path.read_text(encoding='utf-8'))
    old_path=ROOT/'configs/quality_exclusions.json';old=json.loads(old_path.read_text(encoding='utf-8'))
    baseline={r['id']:r for r in read_jsonl(ROOT/'data/prepared/train.jsonl')}
    blocked_pairs={pair_key(baseline[r['id']]) for r in old['excluded']}
    blocked_pairs.update(r['pair_key'] for r in review['reviews'] if r['split'] in {'train','dev'} and r['decision']=='quarantine')
    # Reserve reference reviews do not enter training eligibility decisions.
    dev_blocked={r['id'] for r in review['reviews'] if r['split']=='dev' and r['decision']=='quarantine'}
    original_dev=list(read_jsonl(ROOT/'data/prepared/dev.jsonl'))
    dev=[r for r in original_dev if r['id'] not in dev_blocked and not corruption_flags(r)]
    tests,invalid=evaluation_items();test_index=OverlapIndex(t for r in tests for t in (r['input'],r['reference']))
    held={'北史','北齐书','礼记'};dev_books={'梁书','陈书','容斋随笔'};seen=set();parents=[];excluded=[]
    for path in (ROOT/'assets/source_dataset/train.jsonl',ROOT/'assets/source_dataset/dev.jsonl'):
        for number,row in enumerate(read_jsonl(path),1):
            task=row['task'];source=row['messages'][1]['content'].split('\n',1)[-1].strip();reference=row['messages'][-1]['content'].strip();book=row.get('meta',{}).get('book','')
            item={'id':digest([task,source,reference])[:24],'task':task,'input':source,'reference':reference,
                  'book':book,'instruction':INSTRUCTIONS.get(task,''),'origin':{'file':str(path.relative_to(ROOT)),'record_number':number,'metadata':row.get('meta',{})}}
            reason=None
            if task not in INSTRUCTIONS:reason='unsupported_task'
            elif not book:reason='unknown_book'
            elif book in dev_books|held:reason='dev_or_held_book'
            elif not source or not reference:reason='empty'
            elif pair_key(item) in blocked_pairs:reason='reviewed_pair_quarantine_both_directions'
            elif corruption_flags(item):reason=';'.join(corruption_flags(item))
            elif task=='punctuate' and body(source)!=body(reference):reason='punctuation_body_mismatch'
            elif test_index.matches(source) or test_index.matches(reference):reason='test_overlap'
            elif (task,body(source)) in seen:reason='duplicate_input'
            elif task!='punctuate' and translation_flags(item):reason=';'.join(translation_flags(item))
            if reason:excluded.append({'id':item['id'],'reason':reason,'origin':item['origin']});continue
            seen.add((task,body(source)));parents.append(item)
    dev_index=OverlapIndex(t for r in dev for t in (r['input'],r['reference']))
    rng=random.Random(args.seed);rng.shuffle(parents);pools=defaultdict(list)
    for parent in parents:
        if dev_index.matches(parent['input']) or dev_index.matches(parent['reference']):
            excluded.append({'id':parent['id'],'reason':'fixed_dev_overlap'});continue
        if parent['task']=='punctuate':
            children=punctuation_windows(parent);rng.shuffle(children);options=children
        else:options=[parent]
        for item in options:
            if test_index.matches(item['input']) or test_index.matches(item['reference']):continue
            try:encoded=tokenize_record(tokenizer,item)
            except OverflowError:continue
            pools[item['task']].append((item,encoded));break # At most one child per parent.
    shares={'punctuate':.35,'c2m':.38,'m2c':.27};counts=Counter();chosen=[]
    averages={t:sum(x[1]['supervised_tokens'] for x in p)/len(p) for t,p in pools.items()}
    for _ in range(args.samples):
        available=[t for t in shares if pools[t]]
        if not available:raise ValueError('Not enough eligible training data')
        task=min(available,key=lambda t:(counts[t]+averages[t])/shares[t]);pair=pools[task].pop();chosen.append(pair);counts[task]+=pair[1]['supervised_tokens']
    rng.shuffle(chosen);encoded_dev=[tokenize_record(tokenizer,r) for r in dev]
    task_dev=Counter(r['task'] for r in dev)
    if any(task_dev[t]<12 for t in INSTRUCTIONS):raise ValueError('Insufficient fixed reviewed dev coverage')
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train',[p[0] for p in chosen]),('train-tokenized',[p[1] for p in chosen]),
        ('dev',dev),('dev-tokenized',encoded_dev),('test',tests),('excluded',excluded),('invalid-test-references',invalid)]:write_jsonl(output/(name+'.jsonl'),rows)
    manifest={'recipe':'reviewed_v3','seed':args.seed,'samples':len(chosen),'dev_samples':len(dev),'test_samples':len(tests),
        'tasks':dict(Counter(p[0]['task'] for p in chosen)),'effective_supervised_tokens':dict(counts),
        'supervised_token_share':{k:v/sum(counts.values()) for k,v in counts.items()},
        'dev_tasks':dict(task_dev),'dev_policy':'Keep original fixed dev IDs except model-blind assistant quarantine and character corruption',
        'dev_books':sorted(dev_books),'held_out_books':sorted(held),'train_books':sorted({p[0]['book'] for p in chosen}),
        'punctuation_unique_parents':len({p[0]['parent_id'] for p in chosen if p[0]['task']=='punctuate'}),
        'punctuation_window_policy':{'target':80,'minimum':40,'maximum_soft':120,'children_per_parent_max':1,'synthesized_punctuation':False},
        'excluded':dict(Counter(r['reason'] for r in excluded)),
        'rules_caution':'Eligibility/quarantine rules and assistant text review are not a corpus-wide semantic certification or human blind evaluation',
        'reviews_used_for_training':['train','dev'],'reserve_reviews_used_for_training':False,
        'review_sha256':file_digest(review_path),'prior_review_sha256':file_digest(old_path),
        'prepare_code_sha256':file_digest(ROOT/'classical/prepare_reviewed.py'),'quality_code_sha256':file_digest(ROOT/'classical/quality.py'),
        'chat_template_sha256':digest(tokenizer.chat_template),'files':{p.name:file_digest(p) for p in output.glob('*.jsonl')}}
    manifest['fingerprint']=digest(manifest);write_json(output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('samples','tasks','dev_samples','dev_tasks','effective_supervised_tokens','punctuation_unique_parents','fingerprint')},ensure_ascii=False))

if __name__=='__main__':main()
