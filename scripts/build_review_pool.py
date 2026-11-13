"""Blind paired outputs for later human review; never auto-fill human labels."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,read_jsonl,write_json,write_jsonl


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='clean_2k_seed24');args=ap.parse_args()
    output=ROOT/'reports/human-review-pool.jsonl'
    human_fields=['human_preference','meaning_preserved_A','meaning_preserved_B','omissions_or_additions',
                  'proper_names_and_negation','reviewer','reference_alignment','reference_issue']
    if output.exists() and any(row.get(k) is not None for row in read_jsonl(output) for k in human_fields):
        raise FileExistsError('Existing human review labels are preserved; choose a new review batch before regenerating')
    base={x['id']:x for x in read_jsonl(ROOT/'reports/eval-primary-base-samples.jsonl')}
    ft={x['id']:x for x in read_jsonl(ROOT/'reports'/f'eval-primary-{args.model}-samples.jsonl')}
    rng=random.Random(24);ids=[]
    for direction in ('c2m','m2c'):
        pool=[key for key in base if base[key]['task']=='translate' and key in ft
            and ('m2c' if 'm2c' in base[key].get('label','') else 'c2m')==direction]
        rng.shuffle(pool)
        if len(pool)<25:raise ValueError('Insufficient completed paired translations for '+direction)
        ids.extend(pool[:25])
    rng.shuffle(ids)
    rows=[];keys=[]
    for key in ids[:50]:
        first,second=('base',args.model) if rng.randrange(2) else (args.model,'base')
        source={'base':base[key],args.model:ft[key]}
        rows.append({'id':key,'suite':base[key]['suite'],'direction':'m2c' if 'm2c' in base[key].get('label','') else 'c2m',
            'input':base[key]['input'],'reference':base[key]['reference'],
            'output_A':source[first]['text'],'output_B':source[second]['text'],
            'reference_alignment':None,'reference_issue':None,
            'human_preference':None,'meaning_preserved_A':None,'meaning_preserved_B':None,
            'omissions_or_additions':None,'proper_names_and_negation':None,'reviewer':None})
        keys.append({'id':key,'A':first,'B':second})
    write_jsonl(ROOT/'reports/human-review-pool.jsonl',rows);write_json(ROOT/'data/human-review-key.json',keys)
    print('Blind human review rows created:',len(rows),'all human labels remain null.')


if __name__=='__main__':main()
