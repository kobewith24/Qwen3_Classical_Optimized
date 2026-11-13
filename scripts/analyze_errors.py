"""Describe paired changes and source-body failures without judging translation semantics."""
import argparse
import json
import sys
from collections import Counter,defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,file_digest,read_jsonl,write_json
from classical.evaluate import paired_ci


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='clean_2k_seed24')
    ap.add_argument('--before',default='base,legacy');ap.add_argument('--tag',default='primary')
    ap.add_argument('--output-dir',default='reports');args=ap.parse_args()
    output=(ROOT/args.output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Reports must stay in this project')
    report=json.loads((ROOT/'reports'/f'eval-{args.tag}.json').read_text(encoding='utf-8'))
    if 'paired_vs_base' not in report:raise RuntimeError('Wait for the complete frozen-set comparison')
    before_models=args.before.split(',');models=list(dict.fromkeys([*before_models,args.model]));rows={};failures={};comparisons={}
    for model in models:
        path=ROOT/'reports'/f'eval-{args.tag}-{model}-samples.jsonl'
        rows[model]={x['id']:x for x in read_jsonl(path)};failed=[]
        for row in rows[model].values():
            if row['task']!='punctuate' or row['metrics']['body_preserved']:continue
            before,after=body(row['input']),body(row['text']);edits=[]
            for operation,a,b,c,d in SequenceMatcher(None,before,after,autojunk=False).get_opcodes():
                if operation!='equal':edits.append({'operation':operation,'source':before[a:b],'candidate':after[c:d],
                    'source_start':a,'candidate_start':c})
            failed.append({'id':row['id'],'input':row['input'],'candidate':row['text'],'reference':row['reference'],
                           'edits':edits,'character_error_rate':row['metrics']['body_character_error_rate']})
        failures[model]={'count':len(failed),'examples':failed,
            'edit_operation_counts':dict(Counter(e['operation'] for row in failed for e in row['edits']))}
    for before in before_models:
        grouped=defaultdict(list)
        for key,row in rows[args.model].items():
            if key not in rows[before]:raise ValueError('Paired IDs differ')
            grouped[f"{row['suite']}/{row.get('label',row['task'])}"].append(key)
        comparison={}
        for group,ids in grouped.items():
            old={key:rows[before][key]['metrics']['score'] for key in ids}
            new={key:rows[args.model][key]['metrics']['score'] for key in ids}
            delta={key:new[key]-old[key] for key in ids}
            comparison[group]={**paired_ci(old,new),'higher':sum(v>1e-12 for v in delta.values()),
                'same':sum(abs(v)<=1e-12 for v in delta.values()),'lower':sum(v< -1e-12 for v in delta.values())}
        comparisons[args.model+' minus '+before]=comparison
    result={'paired_score_changes':comparisons,'punctuation_body_failures':failures,
        'selection_sha256':report['selection_sha256'],
        'source_sample_files':{m:file_digest(ROOT/'reports'/f'eval-{args.tag}-{m}-samples.jsonl') for m in models},
        'translation_caution':'chrF differences and wins are reference-overlap screening, not human semantic correctness.'}
    write_json(output/'error-analysis.json',result)
    lines=['# 冻结题集错误分析','','比较对象使用完全相同的题目、提示词和生成上限。断句严格分数在改字时为零；翻译得分为 chrF，不能当语义准确率。','',
        '|比较|题集|n|平均变化|95% 配对 bootstrap 区间|分数升/平/降|','|---|---|---:|---:|---|---|']
    for name,groups in comparisons.items():
        for group,value in groups.items():
            lo,hi=value['bootstrap_95_ci']
            lines.append(f"|{name}|{group}|{value['n']}|{value['mean_delta']:+.4f}|[{lo:+.4f}, {hi:+.4f}]|{value['higher']} / {value['same']} / {value['lower']}|")
    lines+=['','区间仅反映当前样本重采样，不包含训练种子、典籍选择或配方选择的变异。区间跨零的变化不应直接写成稳定提升。','',
        '## 正文改动','','|模型|有正文改动的断句样本|编辑片段类型计数|','|---|---:|---|']
    for name,value in failures.items():lines.append(f"|{name}|{value['count']}|{value['edit_operation_counts']}|")
    lines+=['','全部改字实例、原文/模型候选/参考答案及差异位置见 error-analysis.json。界面回退和人工校对不参与上述模型评分。','',
        '## 后续取证','','翻译的专名、否定、增漏译和跨段指代需由人工填写 human-review-pool.jsonl；当前人工标签为空。保留尚未使用的题目作为下一轮最终测试，不能反复看本轮结果调参后仍把它当完全未见测试。']
    (output/'ERROR_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Paired error analysis saved; translation semantics remain unreviewed.')


if __name__=='__main__':main()
