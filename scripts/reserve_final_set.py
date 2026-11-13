"""Reserve unscored, disjoint IDs for a future recipe; generates no model answers."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,file_digest,read_jsonl,write_json
from classical.prepare import OverlapIndex


def main():
    output=ROOT/'reports/eval-final-reserve-selection.json'
    if output.exists():raise FileExistsError('Reserved IDs are already frozen; do not resample after inspecting results')
    primary=json.loads((ROOT/'reports/eval-primary-selection.frozen.json').read_text(encoding='utf-8'))
    used=set(primary['ids']);legacy=list(read_jsonl(ROOT/'assets/source_dataset/train.jsonl'))[:2000]
    exposure=OverlapIndex(t for x in legacy for t in (x['messages'][1]['content'].split('\n',1)[-1],x['messages'][-1]['content']))
    groups=defaultdict(list);excluded=[]
    for item in read_jsonl(ROOT/'data/prepared/test.jsonl'):
        if item['task'] not in {'translate','punctuate'}:continue
        if exposure.matches(item['input']) or exposure.matches(item['reference']):excluded.append(item['id']);continue
        if item['id'] not in used:groups[(item['suite'],item.get('label',item['task']))].append(item)
    rng=random.Random(20261113);ids=[]
    for group,rows in sorted(groups.items()):
        if len(rows)<30:raise ValueError('Insufficient unscored rows for '+str(group))
        rng.shuffle(rows);ids.extend(x['id'] for x in rows[:30])
    assert len(ids)==150 and not used.intersection(ids)
    write_json(output,{'ids':ids,'sampling_seed':20261113,'limit_per_label':30,'tasks':'translate,punctuate',
        'test_sha256':file_digest(ROOT/'data/prepared/test.jsonl'),
        'excluded_for_legacy_exposure':excluded,'legacy_exposure_policy':'First saved 2k reconstructed exposure; historical run lacks its data hash',
        'primary_selection_sha256':file_digest(ROOT/'reports/eval-primary-selection.frozen.json'),
        'status':'reserved_only_no_model_generations','reference_semantic_review_complete':False,
        'purpose':'Future recipe validation after reference review; disjoint from 200 scored primary IDs'})
    print('Reserved 150 unscored IDs, disjoint from primary; no model loaded or outputs generated.')


if __name__=='__main__':main()
