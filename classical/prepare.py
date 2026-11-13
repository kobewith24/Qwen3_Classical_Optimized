"""Freeze clean, book-disjoint splits before training; exclude benchmark overlap."""
import argparse
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .core import ROOT, INSTRUCTIONS, body, digest, file_digest, read_jsonl, tokenize_record, write_json, write_jsonl


class OverlapIndex:
    def __init__(self, texts=(), anchor=12):
        self.anchor = anchor; self.texts = []; self.index = defaultdict(set); self.exact = set()
        for text in texts: self.add(text)

    def add(self, text):
        key = body(text)
        if key: self.exact.add(key)
        if len(key) < 20: return
        idx = len(self.texts); self.texts.append(key)
        for i in range(len(key) - self.anchor + 1): self.index[key[i:i+self.anchor]].add(idx)

    def matches(self, text):
        key = body(text)
        if key and key in self.exact: return True
        if len(key) < 20: return False
        candidates = set()
        for i in range(len(key) - self.anchor + 1): candidates.update(self.index.get(key[i:i+self.anchor], ()))
        for idx in candidates:
            other = self.texts[idx]
            if key in other or other in key: return True
            if min(len(key), len(other)) / max(len(key), len(other)) >= .8:
                if SequenceMatcher(None, key, other, autojunk=False).ratio() >= .9: return True
        return False


def evaluation_items():
    items = []; invalid = []
    for path in sorted((ROOT/'assets/bench').glob('*.jsonl')):
        for n, row in enumerate(read_jsonl(path)):
            item = {'id': f'bench/{path.stem}/{n}', 'suite': 'bench', 'task': path.stem,
                    'input': row['input'], 'instruction': row.get('instruction', ''),
                    'reference': row['reference'], 'metadata': row.get('metadata') or {}}
            if item['task'] == 'punctuate' and body(item['input']) != body(item['reference']):
                invalid.append(item); continue
            items.append(item)
    for path in sorted((ROOT/'assets/source_dataset/ood').glob('*.jsonl')):
        for n, row in enumerate(read_jsonl(path)):
            direction = 'c2m' if 'c2m' in path.stem else 'm2c' if 'm2c' in path.stem else 'punctuate'
            item = {**row, 'id': f'ood/{path.stem}/{n}', 'suite': 'ood', 'label': path.stem,
                    'task': row.get('task', 'translate'), 'instruction': row.get('instruction') or INSTRUCTIONS[direction]}
            if item['task'] == 'punctuate' and body(item['input']) != body(item['reference']):
                invalid.append(item); continue
            items.append(item)
    return items, invalid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--output', default='data/prepared', help='New project-relative directory; existing frozen data is never overwritten')
    ap.add_argument('--quality-exclusions', default='configs/quality_exclusions.json')
    ap.add_argument('--seed', type=int, default=24); args = ap.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Prepared output must stay inside this project')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Frozen output already exists. Reuse it or choose a new --output directory; do not overwrite data referenced by trained weights.')
    review_path=(ROOT/args.quality_exclusions).resolve()
    reviews=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {'excluded':[]}
    rejected={x['id']:x['reason'] for x in reviews['excluded']}
    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'assets/base_model', local_files_only=True)
    tests, invalid = evaluation_items()
    test_index = OverlapIndex(t for x in tests for t in (x['input'], x['reference']))
    held_books = {'北史', '北齐书', '礼记'}; dev_books = {'梁书', '陈书', '容斋随笔'}
    candidates = []; excluded = []; seen = set()
    for path in [ROOT/'assets/source_dataset/train.jsonl', ROOT/'assets/source_dataset/dev.jsonl']:
        for record_number,row in enumerate(read_jsonl(path),1):
            task = row['task']; source = row['messages'][1]['content'].split('\n', 1)[-1].strip()
            reference = row['messages'][-1]['content'].strip(); book = row.get('meta', {}).get('book', '')
            item = {'id': digest([task, source, reference])[:24], 'task': task, 'input': source,
                    'reference': reference, 'book': book, 'instruction': INSTRUCTIONS.get(task, ''),
                    'origin':{'file':str(path.relative_to(ROOT)),'record_number':record_number,'metadata':row.get('meta',{})}}
            reason = None
            if task not in INSTRUCTIONS: reason = 'unsupported_source_task_removed'
            elif item['id'] in rejected: reason = 'reviewed_semantic_conflict'
            elif book in held_books: reason = 'held_out_book'
            elif not source or not reference: reason = 'empty'
            elif task == 'punctuate' and body(source) != body(reference): reason = 'punctuation_body_mismatch'
            elif test_index.matches(source) or test_index.matches(reference): reason = 'benchmark_overlap_exact_or_anchored_near'
            elif (task, body(source)) in seen: reason = 'duplicate_input'
            if reason:
                excluded.append({'id': item['id'], 'task': task, 'reason': reason,
                    'review_note':rejected.get(item['id']),'origin':item['origin']}); continue
            seen.add((task, body(source))); candidates.append(item)
    dev_pool = [x for x in candidates if x['book'] in dev_books]
    dev_index = OverlapIndex(t for x in dev_pool for t in (x['input'], x['reference']))
    train_pool = []
    for item in candidates:
        if item['book'] in dev_books: continue
        if dev_index.matches(item['input']) or dev_index.matches(item['reference']):
            excluded.append({'id': item['id'], 'task': item['task'], 'reason': 'dev_text_overlap'}); continue
        train_pool.append(item)
    rng.shuffle(train_pool); rng.shuffle(dev_pool)
    pools = defaultdict(list); dev_pools = defaultdict(list)
    for collection, destination in [(train_pool, pools), (dev_pool, dev_pools)]:
        for item in collection:
            try: encoded = tokenize_record(tokenizer, item)
            except OverflowError:
                excluded.append({'id': item['id'], 'task': item['task'], 'reason': 'over_token_budget'}); continue
            destination[item['task']].append((item, encoded))
    shares = {'punctuate': .35, 'c2m': .38, 'm2c': .27}; token_counts = Counter()
    averages = {t: sum(x[1]['supervised_tokens'] for x in p)/len(p) for t,p in pools.items()}
    chosen = []
    for _ in range(args.samples):
        available = [t for t in shares if pools[t]]
        if not available: raise ValueError('Insufficient clean data')
        task = min(available, key=lambda t: (token_counts[t]+averages[t])/shares[t])
        item, encoded = pools[task].pop(); token_counts[task] += encoded['supervised_tokens']; chosen.append((item, encoded))
    rng.shuffle(chosen)
    dev = []
    for task in INSTRUCTIONS:
        assert len(dev_pools[task]) >= 24, f'Not enough dev data for {task}'
        dev.extend(dev_pools[task][:32])
    rng.shuffle(dev)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output/'train.jsonl', [x[0] for x in chosen]); write_jsonl(output/'train-tokenized.jsonl', [x[1] for x in chosen])
    write_jsonl(output/'dev.jsonl', [x[0] for x in dev]); write_jsonl(output/'dev-tokenized.jsonl', [x[1] for x in dev])
    write_jsonl(output/'test.jsonl', tests); write_jsonl(output/'invalid-test-references.jsonl', invalid)
    write_jsonl(output/'excluded.jsonl', excluded)
    hashes = {p.name: file_digest(p) for p in output.glob('*.jsonl')}
    token_total = sum(token_counts.values())
    manifest = {'seed': args.seed, 'samples': len(chosen), 'dev_samples': len(dev), 'test_samples': len(tests),
                'held_out_books': sorted(held_books), 'dev_books': sorted(dev_books),
                'train_books': sorted({x[0]['book'] for x in chosen}),
                'tasks': dict(Counter(x[0]['task'] for x in chosen)), 'effective_supervised_tokens': dict(token_counts),
                'supervised_token_share': {t: n/token_total for t,n in token_counts.items()},
                'excluded': dict(Counter(x['reason'] for x in excluded)), 'invalid_test_references': len(invalid),
                'overlap_policy': 'remove punctuation/space; exact match at any nonempty length; >=20-character containment or shared 12-character anchor plus SequenceMatcher>=0.90; not a semantic guarantee',
                'prepare_code_sha256': file_digest(ROOT/'classical/prepare.py'),
                'quality_review_sha256':file_digest(review_path) if review_path.exists() else None,
                'quality_review_scope':reviews.get('scope','No semantic review configured'),
                'model_config_sha256': file_digest(ROOT/'assets/base_model/config.json'),
                'chat_template_sha256': digest(tokenizer.chat_template), 'files': hashes}
    manifest['fingerprint'] = digest(manifest); write_json(output/'manifest.json', manifest)
    print(manifest, flush=True)


if __name__ == '__main__': main()
