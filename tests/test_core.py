import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classical.core import ROOT,body,file_digest,messages,punctuation_scores,read_jsonl
from classical.evaluate import AnswerCache
from classical.prepare import OverlapIndex,main as prepare_main
from classical.reader import segments,review


class MetricsTests(unittest.TestCase):
    def test_wrong_characters_cannot_get_full_punctuation_credit(self):
        result=punctuation_scores('春眠不觉','甲乙，丙丁。','春眠，不觉。')
        self.assertEqual(result['boundary_f1'],1)
        self.assertEqual(result['score'],0)
        self.assertFalse(result['body_preserved'])

    def test_punctuation_type_and_boundary_are_distinct(self):
        result=punctuation_scores('春眠不觉','春眠。不觉。','春眠，不觉。')
        self.assertEqual(result['score'],1)
        self.assertLess(result['typed_f1'],1)

    def test_invalid_gold_is_rejected(self):
        with self.assertRaises(ValueError):punctuation_scores('春眠不觉','春眠，不觉。','春眠，不知。')

    def test_character_error_rate_counts_deletions(self):
        result=punctuation_scores('春眠不觉','春眠，不。','春眠，不觉。')
        self.assertEqual(result['body_character_error_rate'],.25)
        self.assertEqual(result['score'],0)

    def test_unicode_and_whitespace_do_not_shift_boundaries(self):
        result=punctuation_scores('𠀀春眠\n不觉','𠀀春眠，不觉。','𠀀春眠，不觉。')
        self.assertEqual(result['score'],1)


class DatasetTests(unittest.TestCase):
    def test_quality_candidate_blocks_reviewed_conflicts_without_changing_baseline(self):
        reviews=json.loads((ROOT/'configs/quality_exclusions.json').read_text(encoding='utf-8'))
        blocked={r['id'] for r in reviews['excluded']}
        baseline={r['id'] for r in read_jsonl(ROOT/'data/prepared/train.jsonl')}
        candidate=list(read_jsonl(ROOT/'data/quality_candidate_v2/train.jsonl'))
        self.assertTrue(blocked.issubset(baseline))
        self.assertTrue(blocked.isdisjoint({r['id'] for r in candidate}))
        pair=lambda r:tuple(sorted([body(r['input']),body(r['reference'])]))
        original={r['id']:r for r in read_jsonl(ROOT/'data/prepared/train.jsonl')}
        conflict_pairs={pair(original[key]) for key in blocked}
        self.assertTrue(conflict_pairs.isdisjoint({pair(r) for r in candidate}))
        self.assertEqual(len(candidate),2000)
        self.assertTrue(all(r['origin']['record_number']>0 for r in candidate))
        trained=json.loads((ROOT/'reports/train-clean_2k_seed24.json').read_text(encoding='utf-8'))
        self.assertEqual(file_digest(ROOT/'data/prepared/manifest.json'),trained['dataset_manifest_sha256'])

    def test_existing_frozen_dataset_is_not_overwritten(self):
        path=ROOT/'data/prepared/manifest.json';before=file_digest(path)
        with patch('sys.argv',['prepare']):
            with self.assertRaises(FileExistsError):prepare_main()
        self.assertEqual(file_digest(path),before)

    def test_book_split_and_supervised_labels(self):
        train=list(read_jsonl(ROOT/'data/prepared/train.jsonl'));dev=list(read_jsonl(ROOT/'data/prepared/dev.jsonl'))
        self.assertEqual(len(train),2000)
        self.assertTrue({x['book'] for x in train}.isdisjoint({x['book'] for x in dev}))
        self.assertTrue({x['book'] for x in train}.isdisjoint({'北史','北齐书','礼记'}))
        self.assertNotIn('source',{x['task'] for x in train})
        for row in read_jsonl(ROOT/'data/prepared/train-tokenized.jsonl'):
            n=row['prompt_tokens'];self.assertGreater(n,0)
            self.assertEqual(row['labels'][:n],[-100]*n)
            self.assertEqual(row['labels'][n:],row['input_ids'][n:])
            self.assertGreater(len(row['labels'][n:]),0)

    def test_no_exact_test_input_or_answer_in_training(self):
        texts={body(t) for path in ['train.jsonl','dev.jsonl'] for row in read_jsonl(ROOT/'data/prepared'/path) for t in [row['input'],row['reference']] if body(t)}
        for row in read_jsonl(ROOT/'data/prepared/test.jsonl'):
            for text in [row['input'],row['reference']]:
                if body(text):self.assertNotIn(body(text),texts,row['id'])

    def test_short_exact_matches_are_excluded_without_short_near_matching(self):
        idx=OverlapIndex(['学而时习之不亦说乎'])
        self.assertTrue(idx.matches('学而时习之，不亦说乎？'))
        self.assertFalse(idx.matches('学而时习之不亦悦乎'))
        self.assertFalse(idx.matches(''))

    def test_overlap_rule_finds_long_containment_and_anchored_near_match(self):
        text='昔者庄周梦为胡蝶栩栩然胡蝶也自喻适志与不知周也俄然觉则蘧蘧然周也'
        idx=OverlapIndex([text])
        self.assertTrue(idx.matches('前言'+text+'后记'))
        self.assertTrue(idx.matches(text.replace('昔者','昔日')))
        self.assertFalse(idx.matches('春眠不觉晓处处闻啼鸟夜来风雨声花落知多少'))

    def test_generation_cache_changes_with_model_prompt_or_reference(self):
        item={'id':'x','task':'c2m','input':'春眠不觉晓','reference':'春天睡醒不知天亮。'}
        config={'greedy':True,'max_new_tokens':256}
        first=AnswerCache.key('model-a',item,config)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            cache=AnswerCache(Path(directory)/'answers.jsonl');cache.put({'key':first,'text':'answer'})
            reloaded=AnswerCache(cache.path);self.assertIn(first,reloaded.rows)
            self.assertNotIn(AnswerCache.key('model-b',item,config),reloaded.rows)
            self.assertNotIn(AnswerCache.key('model-a',{**item,'input':'另一个问题'},config),reloaded.rows)
            self.assertNotIn(AnswerCache.key('model-a',{**item,'reference':'新参考'},config),reloaded.rows)


class ReaderTests(unittest.TestCase):
    def test_chunk_offsets_reconstruct_every_character(self):
        text='𠀀春眠不觉晓。\n\n处处闻啼鸟。夜来风雨声。'*100
        parts=segments(text,37)
        self.assertEqual(''.join(p['original'] for p in parts),text)
        for p in parts:self.assertEqual(text[p['start']:p['end']],p['original'])

    def test_rejected_body_change_keeps_original(self):
        result=review('punctuate','春眠不觉','甲乙，丙丁。')
        self.assertEqual(result['text'],'春眠不觉')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'],'review_required')

    def test_translation_is_never_marked_human_verified(self):
        self.assertEqual(review('c2m','春眠不觉晓','春天睡醒不知天亮。')['status'],'review_required')

    def test_review_exposes_body_edits_without_changing_exported_source(self):
        result=review('punctuate','春眠不觉晓','春眠，不晓。')
        self.assertEqual(result['text'],'春眠不觉晓')
        self.assertEqual(result['body_differences'],[{'operation':'delete','start':3,'end':4,'source':'觉','candidate':''}])


if __name__=='__main__':unittest.main()
