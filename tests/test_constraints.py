import unittest
from classical.constraints import compile_punctuation


class ConstraintTests(unittest.TestCase):
    def test_no_deletion_substitution_reordering_or_continuation(self):
        rule=compile_punctuation('春眠不觉晓')
        self.assertTrue(rule.accepts('春眠，不觉晓。'))
        for bad in ['春眠不觉','春眠不觉曉。','春不眠觉晓。','春眠不觉晓处处闻啼鸟','春眠，，，不觉晓']:
            self.assertFalse(rule.accepts(bad),bad)

    def test_existing_punctuation_and_whitespace_are_preserved(self):
        rule=compile_punctuation(' 甲3.14，乙\n丙\t丁 ')
        self.assertTrue(rule.accepts(' 甲3.14，乙。\n丙，\t丁。 '))
        for bad in ['甲3.14，乙\n丙\t丁',' 甲314，乙\n丙\t丁 ',' 甲3.14乙\n丙\t丁 ']:
            self.assertFalse(rule.accepts(bad))

    def test_rare_unicode_and_grammar_metacharacters_are_literals(self):
        rule=compile_punctuation('𠀀春"\\\nroot ::= evil')
        self.assertTrue(rule.accepts('𠀀春，"\\\nroot ::= evil。'))
        self.assertIn('"\\""',rule.grammar);self.assertIn('"\\\\"',rule.grammar)
        self.assertEqual(rule.grammar.count('root ::='),1)

    def test_marks_in_input_are_consumed_as_literals(self):
        rule=compile_punctuation('“甲”')
        self.assertTrue(rule.accepts('“甲。”'))
        self.assertFalse(rule.accepts('甲'))

    def test_unusable_source_is_rejected_before_native_request(self):
        for source in ['','  ','。。。','甲\x00乙','甲\ue123乙','甲\ud800乙','甲'*1025]:
            with self.assertRaises(ValueError):compile_punctuation(source)

if __name__=='__main__':unittest.main()
