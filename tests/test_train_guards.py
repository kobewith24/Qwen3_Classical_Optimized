import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from classical.core import ROOT,file_digest
from classical.train import main


class TrainingGuardTests(unittest.TestCase):
    def test_candidate_preflight_does_not_create_model_outputs(self):
        name='guard-preflight-24';captured=io.StringIO()
        with patch('sys.argv',['train','--name',name,'--data-dir','data/quality_candidate_v2','--validate-only']),redirect_stdout(captured):main()
        result=json.loads(captured.getvalue());self.assertTrue(result['validated']);self.assertFalse(result['model_loaded'])
        self.assertEqual(result['samples'],2000);self.assertFalse((ROOT/'weights'/name).exists())

    def test_existing_release_is_preserved_on_accidental_rerun(self):
        report=ROOT/'reports/train-clean_2k_seed24.json';before=file_digest(report)
        with patch('sys.argv',['train','--name','clean_2k_seed24']):
            with self.assertRaises(FileExistsError):main()
        self.assertEqual(file_digest(report),before)

    def test_experiment_name_cannot_escape_the_project(self):
        with patch('sys.argv',['train','--name','../outside']):
            with self.assertRaises(ValueError):main()

    def test_data_path_cannot_use_original_project(self):
        with patch('sys.argv',['train','--name','guard-test','--data-dir',str(ROOT.parent/'Qwen3-4B-Classical')]):
            with self.assertRaises(ValueError):main()

    def test_changed_frozen_labels_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            data=Path(directory);(data/'train-tokenized.jsonl').write_text('{}\n',encoding='utf-8')
            (data/'manifest.json').write_text(json.dumps({'files':{'train-tokenized.jsonl':'invalid-hash'}}),encoding='utf-8')
            with patch('sys.argv',['train','--name','guard-test','--data-dir',str(data)]):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):main()


if __name__=='__main__':unittest.main()
