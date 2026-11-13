from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,body,file_digest,read_jsonl,write_json

ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='data/prepared');ap.add_argument('--report',default='reports/label-audit.json');args=ap.parse_args()
data=(ROOT/args.data_dir).resolve();report=(ROOT/args.report).resolve()
assert data.is_relative_to(ROOT.resolve()) and report.is_relative_to(ROOT.resolve())
manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
assert all(file_digest(data/name)==value for name,value in manifest['files'].items())
rows=list(read_jsonl(data/'train-tokenized.jsonl'));lengths=sorted(len(x['input_ids']) for x in rows)
test_texts={body(x[k]) for x in read_jsonl(data/'test.jsonl') for k in ('input','reference') if body(x[k])}
overlaps=[{'id':x['id'],'field':k,'split':split} for split in ('train','dev')
    for x in read_jsonl(data/f'{split}.jsonl') for k in ('input','reference') if body(x[k]) in test_texts]
assert not overlaps,overlaps
audit={'samples':len(rows),'effective_supervised_tokens':sum(x['supervised_tokens'] for x in rows),
    'scope':'Token labels, file hashes, lengths and exact test overlap only; NOT semantic reference correctness',
    'max_total_tokens':max(lengths),'p95_total_tokens':lengths[int(len(lengths)*.95)],'p99_total_tokens':lengths[int(len(lengths)*.99)],
    'all_prompt_labels_masked':all(x['labels'][:x['prompt_tokens']]==[-100]*x['prompt_tokens'] for x in rows),
    'all_answer_labels_present':all(x['labels'][x['prompt_tokens']:]==x['input_ids'][x['prompt_tokens']:] for x in rows),
    'silent_truncation':False,'prepared_files_verified':True,'all_length_exact_train_dev_test_overlaps':overlaps,
    'manifest_fingerprint':manifest['fingerprint']}
write_json(report,audit);print(audit)
