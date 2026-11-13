import unittest
from classical.evaluate import task_names


class EvaluationTaskTests(unittest.TestCase):
    def test_development_translation_tasks_are_not_silently_skipped(self):
        self.assertEqual(task_names('dev'),{'c2m','m2c','punctuate'})
        self.assertEqual(task_names('test'),{'translate','punctuate'})
        self.assertEqual(task_names('dev','punctuate'),{'punctuate'})

if __name__=='__main__':unittest.main()
