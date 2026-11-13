import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from classical.core import ROOT
from classical.merge import merged_tensor


class MergeTests(unittest.TestCase):
    def test_base_key_mapping_and_lora_scaling(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            path=Path(directory)/'adapter.safetensors'
            save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':torch.tensor([[1.,0.]]),
                       'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':torch.tensor([[1.],[0.]])},str(path))
            with safe_open(path,framework='pt',device='cpu') as adapter:
                result,changed=merged_tensor(torch.tensor([[1.,2.],[3.,4.]]),'model.layers.0.self_attn.q_proj.weight',adapter,{'r':1,'lora_alpha':2})
            self.assertTrue(changed)
            self.assertTrue(torch.equal(result,torch.tensor([[3.,2.],[3.,4.]],dtype=torch.float16)))

    def test_unadapted_tensor_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            path=Path(directory)/'adapter.safetensors';save_file({'dummy':torch.ones(1)},str(path))
            with safe_open(path,framework='pt',device='cpu') as adapter:
                result,changed=merged_tensor(torch.tensor([1.,2.]),'model.norm.weight',adapter,{'r':1,'lora_alpha':2})
            self.assertFalse(changed);self.assertTrue(torch.equal(result,torch.tensor([1.,2.],dtype=torch.float16)))


if __name__=='__main__':unittest.main()
