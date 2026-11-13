"""Descriptive two-seed comparison; never choose the default from test scores."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,file_digest,read_jsonl,write_json
from classical.evaluate import paired_ci


def main():
    names=['clean_2k_seed24','clean_2k_seed42']
    train={n:json.loads((ROOT/'reports'/f'train-{n}.json').read_text(encoding='utf-8')) for n in names}
    keys=['epochs','samples','tasks','effective_supervised_tokens','warmup_ratio','lr','lora_rank',
          'dataset_manifest_sha256','training_source_sha256']
    for key in keys:
        if train[names[0]][key]!=train[names[1]][key]:raise ValueError('Recipes differ beyond seed: '+key)
    if not all(t['initialization_seed_controlled'] for t in train.values()):raise ValueError('Uncontrolled initialization')
    if train[names[0]]['adapter_sha256']==train[names[1]]['adapter_sha256']:raise ValueError('Duplicate adapter weights')
    report=json.loads((ROOT/'reports/eval-primary.json').read_text(encoding='utf-8'))
    frozen=file_digest(ROOT/'reports/eval-primary-selection.frozen.json')
    if report['selection_sha256']!=frozen:raise ValueError('Primary selection changed')
    rows={n:{x['id']:x for x in read_jsonl(ROOT/'reports'/f'eval-primary-{n}-samples.jsonl')} for n in names}
    if set(rows[names[0]])!=set(rows[names[1]]):raise ValueError('Seed comparison is not paired')
    result={}
    for group,first in report['models'][names[0]].items():
        second=report['models'][names[1]][group]
        values=[first['score'],second['score']]
        ids=[key for key,row in rows[names[0]].items() if f"{row['suite']}/{row.get('label',row['task'])}"==group]
        result[group]={'n_per_seed':len(ids),'scores':dict(zip(names,values)),'two_seed_mean':sum(values)/2,
            'observed_seed_range':max(values)-min(values),
            'paired_seed42_minus_seed24':paired_ci({key:rows[names[0]][key]['metrics']['score'] for key in ids},
                                                  {key:rows[names[1]][key]['metrics']['score'] for key in ids})}
    write_json(ROOT/'reports/seed-comparison.json',{'groups':result,'same_recipe_except_seed':True,
        'adapter_hashes':{n:t['adapter_sha256'] for n,t in train.items()},'selection_sha256':frozen,
        'default_model':'clean_2k_seed24','default_policy':'Chosen before this repeat; not switched using primary test scores',
        'limitation':'Two seeds only; descriptive range is not a reliable population variance or proof of saturation.'})
    lines=['# 两个训练种子的重复实验','','两次使用同一冻结数据、监督 token 配比、代码、warmup、学习率和训练步数；LoRA 初始化及数据顺序由种子控制，种子在构建 LoRA 之前设置。各自仅用开发损失选择 checkpoint。','',
        '|题集|每个种子 n|seed24|seed42|两次均值|观察到的极差|','|---|---:|---:|---:|---:|---:|']
    for group,value in result.items():
        a,b=value['scores'].values()
        lines.append(f"|{group}|{value['n_per_seed']}|{a:.4f}|{b:.4f}|{value['two_seed_mean']:.4f}|{value['observed_seed_range']:.4f}|")
    lines+=['','断句列为严格边界 F1 均值，翻译列为归一到 0–1 的逐句 chrF 均值，不是语义准确率。',
        '默认服务仍用预先确定的 seed24，没有依据本轮测试分数更换种子。配对样本区间见 seed-comparison.json，它不等于训练种子分布的置信区间。','',
        '两次重复只能初步暴露波动，不能证明稳定饱和、通用能力保留或翻译语义准确率。pilot 的初始化未预先固定，不算第三个受控种子。']
    (ROOT/'reports/SEED_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Two controlled seeds compared on unchanged primary IDs; default remains seed24.')


if __name__=='__main__':main()
