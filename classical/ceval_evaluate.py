"""Official val subset, zero-shot chat, constrained single-token A-D likelihood."""
import argparse
import contextlib
import json
import math
import re
from collections import defaultdict
from .core import ROOT,digest,file_digest,read_jsonl,write_json,write_jsonl
from .evaluate import paired_ci

SYSTEM='你是一个认真作答的中文助手。请从给出的四个选项中选择唯一正确的选项。'


def turns(item):
    text=f"以下是中国关于{item['subject_zh']}考试的单项选择题，请选出其中的正确答案。\n\n{item['question']}\n"
    text+='\n'.join(k+'. '+item['choices'][k] for k in 'ABCD')+'\n答案：'
    return [{'role':'system','content':SYSTEM},{'role':'user','content':text}]


def choice_from_logits(values):
    if len(values)!=4:raise ValueError('Exactly four A-D logits required')
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite choice logits')
    return 'ABCD'[max(range(4),key=lambda i:values[i])]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',default='base,clean_2k_seed24')
    ap.add_argument('--tag',default='ceval-val-20261113');ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--max-input-tokens',type=int,default=2048);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):raise ValueError('Invalid report tag')
    names=[n.strip() for n in args.models.split(',')]
    if len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',n) for n in names):raise ValueError('Invalid model names')
    if args.batch_size<1 or args.max_input_tokens<64:raise ValueError('Invalid batch/context settings')
    folder=ROOT/'data/ceval_val_260';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if file_digest(folder/'questions.jsonl')!=manifest['questions_sha256']:raise ValueError('Frozen C-Eval sample changed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model',local_files_only=True);tokenizer.padding_side='left'
    option_tokens=[tokenizer.encode(k,add_special_tokens=False) for k in 'ABCD']
    if any(len(x)!=1 for x in option_tokens):raise ValueError('A-D must each be a single token')
    option_ids=[x[0] for x in option_tokens];items=[];excluded=[]
    for item in read_jsonl(folder/'questions.jsonl'):
        if not item['question'].strip() or any(not str(item['choices'][k]).strip() for k in 'ABCD'):raise ValueError('Empty question or option')
        prompt=tokenizer.apply_chat_template(turns(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        for letter,token in zip('ABCD',option_ids):
            if tokenizer.encode(prompt+letter,add_special_tokens=False)!=ids+[token]:raise ValueError('Answer token boundary is not aligned')
        if len(ids)>args.max_input_tokens:excluded.append({'id':item['id'],'tokens':len(ids),'reason':'over_context_budget_no_truncation'});continue
        items.append({**item,'prompt':prompt,'prompt_tokens':len(ids)})
    write_json(ROOT/'reports'/f'{args.tag}-selection.json',{'ids':[r['id'] for r in items],'excluded':excluded,
        'source_manifest_sha256':file_digest(folder/'manifest.json'),'option_token_ids':dict(zip('ABCD',option_ids)),
        'max_input_tokens':args.max_input_tokens,'option_prefix_alignment_verified':True,'source_samples':manifest['samples']})
    if args.validate_only:
        print(json.dumps({'validated':True,'model_loaded':False,'samples':len(items),'excluded':len(excluded),
            'max_prompt_tokens':max(r['prompt_tokens'] for r in items),'option_token_ids':option_ids},ensure_ascii=False));return
    from .inference import HFEngine
    engine=HFEngine(args.batch_size);torch=engine.torch;tokenizer=engine.tokenizer
    cache_path=ROOT/'cache/ceval-choice-answers.jsonl';cache={x['key']:x for x in read_jsonl(cache_path)} if cache_path.exists() else {}
    method=file_digest(ROOT/'classical/ceval_evaluate.py');results={};all_rows={}
    for name in names:
        path=ROOT/'assets/reference_adapter' if name=='legacy' else ROOT/'weights'/name
        if name!='base' and not (path/'adapter_model.safetensors').exists():raise FileNotFoundError(path)
        engine.select(name,path);fingerprint=engine.fingerprint();rows=[]
        for start in range(0,len(items),args.batch_size):
            batch=items[start:start+args.batch_size];missing=[]
            for item in batch:
                key=digest({'model':fingerprint,'prompt':item['prompt'],'id':item['id'],'answer':item['answer'],
                            'options':option_ids,'method':method,'max_context':args.max_input_tokens})
                if key not in cache:missing.append((key,item))
            if missing:
                enc=tokenizer([r['prompt'] for _,r in missing],padding=True,truncation=False,return_tensors='pt').to(engine.model.device)
                if enc['attention_mask'].sum(-1).tolist()!=[r['prompt_tokens'] for _,r in missing]:raise ValueError('Batched prompt token counts changed')
                mode=engine.model.disable_adapter() if name=='base' and engine.loaded else contextlib.nullcontext()
                with mode,torch.inference_mode():
                    out=engine.model(**enc,use_cache=False,logits_to_keep=1)
                    if out.logits.shape[1]!=1:raise ValueError('Expected last-token logits only')
                    logits=out.logits[:,-1,option_ids].float().cpu().tolist()
                for (key,item),values in zip(missing,logits):
                    value={'key':key,'model_fingerprint':fingerprint,'id':item['id'],'choice_logits':values,
                           'prediction':choice_from_logits(values),'prompt_tokens':item['prompt_tokens']}
                    cache_path.parent.mkdir(parents=True,exist_ok=True)
                    with cache_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
                    cache[key]=value
                del out,enc
            for item in batch:
                key=digest({'model':fingerprint,'prompt':item['prompt'],'id':item['id'],'answer':item['answer'],
                            'options':option_ids,'method':method,'max_context':args.max_input_tokens})
                answer=cache[key];rows.append({k:v for k,v in {**item,**answer,'correct':answer['prediction']==item['answer']}.items() if k!='prompt'})
            write_jsonl(ROOT/'reports'/f'{args.tag}-{name}-samples.jsonl',rows)
            print(name,len(rows),'/',len(items),flush=True)
        grouped=defaultdict(list)
        for row in rows:grouped[row['subject']].append(row)
        categories=defaultdict(list)
        for row in rows:categories[row['category']].append(row)
        rate=lambda rs:{'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs)}
        results[name]={'overall':rate(rows),'subjects':{k:rate(v) for k,v in grouped.items()},
                       'categories':{k:rate(v) for k,v in categories.items()},'model_fingerprint':fingerprint}
        all_rows[name]=rows
        write_json(ROOT/'reports'/f'{args.tag}.json',{'models':results,'complete':False})
    pairs={}
    for before,after in [('base',n) for n in all_rows if n!='base']+[('clean_2k_seed24',n) for n in all_rows if n not in {'base','clean_2k_seed24'}]:
        if before not in all_rows:continue
        pairs[after+' minus '+before]=paired_ci({r['id']:int(r['correct']) for r in all_rows[before]},
                                              {r['id']:int(r['correct']) for r in all_rows[after]})
    report={'models':results,'paired_comparisons':pairs,'complete':True,'samples_per_model':len(items),
        'source_manifest_sha256':file_digest(folder/'manifest.json'),'method_sha256':method,
        'method':'0-shot chat template; compare logits of single-token A/B/C/D; no CoT, no generated-answer regex',
        'scope':'Stratified C-Eval val sample, NOT full test or leaderboard; does not establish no forgetting or general ability retention',
        'source_instructions':manifest['source_instructions'],'data_revision':manifest['revision'],'excluded':excluded}
    write_json(ROOT/'reports'/f'{args.tag}.json',report)
    lines=['# C-Eval val 抽样对照','','官方数据版本固定，每科随机 5 题，共 52 科。0-shot 聊天模板，直接比较 A–D 单 token logits；不做 CoT，不以生成分块数或文字正则猜选项。','',
        '|模型|实际题数|正确数|准确率|','|---|---:|---:|---:|']
    for name,result in results.items():
        value=result['overall'];lines.append(f"|{name}|{value['n']}|{value['correct']}|{value['accuracy']:.4f}|")
    lines+=['',f'共同因长度超预算而剔除 {len(excluded)} 题，无静默截断。配对差值区间、学科/大类结果及原始 logits 见相邻 JSON/JSONL。',
        '这是验证集抽样诊断，不是完整 C-Eval 测试集或官方榜单成绩；区间只覆盖当前抽样，不证明没有遗忘，也不评价自由生成的指令遵从。',
        '[官方数据与评测方法](https://github.com/hkust-nlp/ceval#how-to-evaluate-on-c-eval)']
    (ROOT/'reports'/f'{args.tag}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CEVAL_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
