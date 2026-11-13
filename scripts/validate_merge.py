from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from classical.core import ROOT,write_json

torch.set_num_threads(2)
base=ROOT/'assets/base_model';merged=ROOT/'weights/merged_clean_2k_seed24';adapter=ROOT/'weights/clean_2k_seed24'
old=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
new=json.loads((merged/'model.safetensors.index.json').read_text())['weight_map']
config=json.loads((adapter/'adapter_config.json').read_text())
assert set(old)==set(new)
names=['model.layers.0.self_attn.q_proj.weight','model.layers.17.mlp.down_proj.weight','model.layers.35.self_attn.o_proj.weight','model.norm.weight']
checks=[]
with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as lora:
    for name in names:
        with safe_open(base/old[name],framework='pt',device='cpu') as f:weight=f.get_tensor(name).float()
        prefix='base_model.model.'+name.removesuffix('.weight')
        if prefix+'.lora_A.weight' in lora.keys():
            a=lora.get_tensor(prefix+'.lora_A.weight').float();b=lora.get_tensor(prefix+'.lora_B.weight').float()
            weight.addmm_(b,a,beta=1,alpha=config['lora_alpha']/config['r'])
        expected=weight.half()
        with safe_open(merged/new[name],framework='pt',device='cpu') as f:actual=f.get_tensor(name)
        error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,rtol=.001,atol=.00002)
        checks.append({'tensor':name,'passed':True,'max_absolute_fp16_error':error})
write_json(ROOT/'reports/merge-verification.json',{'same_parameter_keys':True,'parameter_tensors':len(new),'checks':checks})
print('Merged file keys and independently recomputed sampled LoRA updates verified.')
