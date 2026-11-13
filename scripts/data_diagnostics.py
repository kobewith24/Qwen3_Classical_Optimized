"""Descriptive audits of frozen training data; flags are not semantic judgments."""
import json
import statistics
import sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,file_digest,read_jsonl,write_json,write_jsonl


def main():
    rows=list(read_jsonl(ROOT/'data/prepared/train.jsonl'))
    encoded={r['sample_id']:r for r in read_jsonl(ROOT/'data/prepared/train-tokenized.jsonl')}
    task_stats={};pairs=defaultdict(list);by_book=Counter();book_tokens=Counter();flags=[]
    for task in ('punctuate','c2m','m2c'):
        subset=[r for r in rows if r['task']==task]
        input_lengths=[len(body(r['input'])) for r in subset];answer_tokens=[encoded[r['id']]['supervised_tokens'] for r in subset]
        task_stats[task]={'samples':len(subset),'mean_input_body_characters':statistics.mean(input_lengths),
            'median_input_body_characters':statistics.median(input_lengths),'mean_answer_tokens':statistics.mean(answer_tokens),
            'answer_tokens':sum(answer_tokens),'books':len({r['book'] for r in subset})}
    for r in rows:
        by_book[r['book']]+=1;book_tokens[r['book']]+=encoded[r['id']]['supervised_tokens']
        if r['task'] in {'c2m','m2c'}:
            pairs[tuple(sorted([body(r['input']),body(r['reference'])]))].append(r)
        if r['task']=='m2c' and any(w in r['input'] for w in ('不','没有','未曾','不能')):
            if not any(c in r['reference'] for c in '不弗勿毋莫未无非匪罔亡岂'):
                flags.append({**r,'review_reason':'Modern negation cue has no selected classical cue; lexical screening only, NOT a confirmed error',
                              'human_judgment':None})
    bidirectional=[v for v in pairs.values() if {r['task'] for r in v}=={'c2m','m2c'}]
    result={'manifest_sha256':file_digest(ROOT/'data/prepared/manifest.json'),'tasks':task_stats,
        'samples_total':len(rows),'translation_task_records':sum(r['task'] in {'c2m','m2c'} for r in rows),
        'unique_normalized_undirected_translation_pairs':len(pairs),
        'pairs_present_in_both_directions':len(bidirectional),
        'books':[{'book':name,'samples':count,'answer_tokens':book_tokens[name]} for name,count in by_book.most_common()],
        'lexical_negation_review_candidates':len(flags),
        'caution':'Reversed translation records are legitimate two-direction supervision, not 2 independent source pairs. Flags require human judgment; no confirmed label-error rate.'}
    write_json(ROOT/'reports/data-diagnostics.json',result)
    write_jsonl(ROOT/'reports/training-review-candidates.jsonl',flags)
    lines=['# 冻结训练数据诊断','','不改动冻结文件。以下统计用于解释样本多样性与明日数据审查，不是新的测试集结果。','',
        '|任务|样本|平均输入正文字数|平均有效答案 token|有效答案 token 合计|典籍数|','|---|---:|---:|---:|---:|---:|']
    for task,s in task_stats.items():lines.append(f"|{task}|{s['samples']}|{s['mean_input_body_characters']:.1f}|{s['mean_answer_tokens']:.1f}|{s['answer_tokens']}|{s['books']}|")
    lines+=['',f"翻译任务共有 {result['translation_task_records']} 条记录，规范化后为 {len(pairs)} 对不区分方向的文本配对，其中 {len(bidirectional)} 对同时出现在两个方向。双向监督本身合理，但任务条数不能直接称为独立篇章数。",'',
        '|训练典籍（前十）|样本|有效答案 token|','|---|---:|---:|']
    for name,count in by_book.most_common(10):lines.append(f'|{name}|{count}|{book_tokens[name]}|')
    lines+=['',f'另有 {len(flags)} 条今译古记录命中粗略的否定线索筛查，存入 training-review-candidates.jsonl，人工判断为空。词面不匹配可能来自合法的古汉语表达，不能直接剔除或视作错标比例。',
        '此处只审查训练数据。若调整配方，应另存数据版本，不能回填已经查看的测试答案。']
    (ROOT/'reports/DATA_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['translation_task_records','unique_normalized_undirected_translation_pairs',
        'pairs_present_in_both_directions','lexical_negation_review_candidates']},ensure_ascii=False))


if __name__=='__main__':main()
