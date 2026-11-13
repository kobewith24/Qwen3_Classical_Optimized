import json
import math
import unittest
from collections import Counter
from classical.core import ROOT,file_digest,read_jsonl
from classical.ceval_evaluate import turns,choice_from_logits


class CEvalProtocolTests(unittest.TestCase):
    def test_gold_answer_is_not_injected_into_prompt(self):
        item={'subject_zh':'数学','question':'一加一等于多少？','choices':{'A':'1','B':'2','C':'3','D':'4'},'answer':'B'}
        messages=turns(item);text=messages[-1]['content']
        self.assertTrue(text.endswith('答案：'));self.assertNotIn('答案：B',text)
        self.assertNotIn('answer',text)
        self.assertEqual(messages,turns({**item,'answer':'D'}))

    def test_choice_ranking_and_invalid_logits(self):
        self.assertEqual(choice_from_logits([-5,-1,-2,-3]),'B')
        self.assertEqual(choice_from_logits([1,1,1,1]),'A')
        with self.assertRaises(ValueError):choice_from_logits([0,1,2])
        with self.assertRaises(ValueError):choice_from_logits([0,float('nan'),2,3])

    def test_official_sample_is_frozen_and_subject_balanced(self):
        data=ROOT/'data/ceval_val_260';rows=list(read_jsonl(data/'questions.jsonl'))
        manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows),260);self.assertEqual(len({r['id'] for r in rows}),260)
        self.assertEqual(set(Counter(r['subject'] for r in rows).values()),{5})
        self.assertEqual(len(Counter(r['subject'] for r in rows)),52)
        self.assertEqual(file_digest(data/'questions.jsonl'),manifest['questions_sha256'])
        self.assertEqual(manifest['split'],'val')

if __name__=='__main__':unittest.main()
