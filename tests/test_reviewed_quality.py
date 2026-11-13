import json
import unittest
import unicodedata
from classical.core import ROOT,body,read_jsonl
from classical.quality import corruption_flags,pair_key,punctuation_windows,translation_flags


class ReviewedQualityTests(unittest.TestCase):
    def test_private_characters_are_quarantined_but_cjk_extensions_survive(self):
        self.assertIn('private_or_surrogate_character',corruption_flags({'input':'姓名\ue123','reference':'某人'}))
        self.assertEqual(corruption_flags({'input':'𠀀春眠','reference':'𠀀春眠。'}),[])

    def test_pair_quarantine_is_independent_of_translation_direction(self):
        a={'input':'古文甲','reference':'现代文乙'}
        self.assertEqual(pair_key(a),pair_key({'input':a['reference'],'reference':a['input']}))

    def test_copy_windows_preserve_every_selected_body_character(self):
        text=('𠀀春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。'*15)+'末尾'
        item={'id':'parent','task':'punctuate','input':body(text),'reference':text}
        children=punctuation_windows(item)
        self.assertGreater(len(children),1)
        for child in children:
            span=child['body_span'];self.assertEqual(body(text)[span['start']:span['end']],child['input'])
            self.assertEqual(body(child['reference']),child['input'])
            self.assertIn(child['reference'],unicodedata.normalize('NFC',text))

    def test_identity_translation_does_not_count_as_instruction_signal(self):
        row={'task':'m2c','input':'太昌初迁安州刺史累加散骑常侍','reference':'太昌初，迁安州刺史，累加散骑常侍。'}
        self.assertIn('identity_translation_low_information',translation_flags(row))

    def test_reviewed_dataset_keeps_unique_copy_parents_and_holdouts(self):
        data=ROOT/'data/quality_reviewed_v3';train=list(read_jsonl(data/'train.jsonl'));dev=list(read_jsonl(data/'dev.jsonl'))
        punct=[r for r in train if r['task']=='punctuate']
        self.assertEqual(len(train),2000);self.assertEqual(len({r['parent_id'] for r in punct}),len(punct))
        self.assertTrue({r['book'] for r in train}.isdisjoint({r['book'] for r in dev}|{'北史','北齐书','礼记'}))
        self.assertTrue(all(body(r['input'])==body(r['reference']) for r in punct))
        review=json.loads((ROOT/'configs/review_20261113.json').read_text(encoding='utf-8'))
        blocked={r['pair_key'] for r in review['reviews'] if r['split'] in {'train','dev'} and r['decision']=='quarantine'}
        self.assertTrue(blocked.isdisjoint(pair_key(r) for r in train+dev))
        manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['reserve_reviews_used_for_training'])

    def test_reference_review_is_disjoint_from_primary_and_does_not_fake_humans(self):
        primary=json.loads((ROOT/'reports/eval-primary-selection.frozen.json').read_text(encoding='utf-8'))
        final=json.loads((ROOT/'reports/eval-reviewed-final-20261113-selection.json').read_text(encoding='utf-8'))
        self.assertEqual(len(final['ids']),121);self.assertTrue(set(final['ids']).isdisjoint(primary['ids']))
        self.assertFalse(final['reference_semantic_review_complete'])
        self.assertTrue(all(r['human_preference'] is None for r in read_jsonl(ROOT/'reports/human-review-pool.jsonl')))


class FrozenPrecisionTests(unittest.TestCase):
    def test_only_frozen_io_is_recast(self):
        import torch
        from classical.training_utils import configure_frozen_io
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed=torch.nn.Embedding(10,4);self.head=torch.nn.Linear(4,10)
                self.norm=torch.nn.LayerNorm(4);self.adapter=torch.nn.Linear(4,4)
                for module in (self.embed,self.head):
                    for param in module.parameters():param.requires_grad=False
            def get_input_embeddings(self):return self.embed
            def get_output_embeddings(self):return self.head
        model=Model();configure_frozen_io(model,'bf16')
        self.assertEqual(model.embed.weight.dtype,torch.bfloat16);self.assertEqual(model.head.weight.dtype,torch.bfloat16)
        self.assertEqual(model.norm.weight.dtype,torch.float32);self.assertEqual(model.adapter.weight.dtype,torch.float32)
        self.assertTrue(model.adapter.weight.requires_grad)
        model.embed.weight.requires_grad=True
        with self.assertRaises(ValueError):configure_frozen_io(model,'bf16')

if __name__=='__main__':unittest.main()
