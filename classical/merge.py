"""CPU streaming merge: bounded shards instead of loading an 8GB base twice."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from .core import ROOT,file_digest,write_json


def merged_tensor(weight,name,adapter,config):
    import torch
    prefix='base_model.model.'+name.removesuffix('.weight')
    a_key=prefix+'.lora_A.weight';b_key=prefix+'.lora_B.weight'
    if a_key not in adapter.keys():return weight.to(torch.float16),False
    a=adapter.get_tensor(a_key).float();b=adapter.get_tensor(b_key).float()
    value=weight.float()+b@a*(config['lora_alpha']/config['r'])
    if not torch.isfinite(value).all() or value.abs().max()>65504:raise ValueError('Non-finite or overflowing merged weight: '+name)
    return value.to(torch.float16),True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='clean_2k_seed24');ap.add_argument('--shard-mib',type=int,default=384)
    args=ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    base=ROOT/'assets/base_model';adapter_path=ROOT/'weights'/args.name
    config=json.loads((adapter_path/'adapter_config.json').read_text(encoding='utf-8'))
    for field in ['use_dora','use_rslora','fan_in_fan_out']:
        if config.get(field):raise ValueError('Unsupported adapter merge option: '+field)
    if config.get('bias','none')!='none' or config.get('modules_to_save') or config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('This streaming merger requires plain, uniform, bias-free LoRA')
    output=(ROOT/'weights'/f'merged_{args.name}').resolve()
    pending=(ROOT/'weights'/f'merge_pending_{args.name}_{int(time.time())}').resolve()
    assert output.is_relative_to(ROOT.resolve()) and pending.is_relative_to(ROOT.resolve())
    if output.exists():raise FileExistsError(output)
    pending.mkdir(parents=True);started=time.time();mapping={};buffer={};size=0;total=0;shards=[];adapted=0
    def flush():
        nonlocal buffer,size
        if not buffer:return
        filename=f'model-{len(shards)+1:05d}.safetensors'
        save_file(buffer,str(pending/filename),metadata={'format':'pt'})
        for key in buffer:mapping[key]=filename
        shards.append(filename);buffer={};size=0;gc.collect()
    with safe_open(adapter_path/'adapter_model.safetensors',framework='pt',device='cpu') as adapter:
        a_keys=[x for x in adapter.keys() if '.lora_A.' in x]
        for shard in sorted(base.glob('model-*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as source:
                for name in source.keys():
                    value,changed=merged_tensor(source.get_tensor(name),name,adapter,config)
                    n=value.numel()*value.element_size()
                    if buffer and size+n>args.shard_mib*1024**2:flush()
                    buffer[name]=value.contiguous();size+=n;total+=n;adapted+=changed
                    if size>=args.shard_mib*1024**2:flush()
            print('MERGED_SHARD',shard.name,'adapted',adapted,flush=True)
        flush()
        if adapted!=len(a_keys):raise ValueError(f'Adapter mapping incomplete: {adapted} / {len(a_keys)}')
    write_json(pending/'model.safetensors.index.json',{'metadata':{'total_size':total},'weight_map':mapping})
    for path in base.iterdir():
        if path.is_file() and path.suffix in {'.json','.txt','.jinja'} and path.name!='model.safetensors.index.json':
            shutil.copy2(path,pending/path.name)
    base_config=json.loads((pending/'config.json').read_text(encoding='utf-8'))
    base_config['dtype' if 'dtype' in base_config else 'torch_dtype']='float16';write_json(pending/'config.json',base_config)
    # Both resolved directories were checked to stay inside this new project.
    pending.rename(output)
    report={'adapter':args.name,'adapter_sha256':file_digest(adapter_path/'adapter_model.safetensors'),
        'output':str(output),'dtype':'float16','adapted_matrices':adapted,'tensors':len(mapping),'shards':len(shards),
        'weights_bytes':total,'seconds':time.time()-started,'policy':'full precision base + LoRA; no merge into NF4 weights',
        'files':{name:file_digest(output/name) for name in shards}}
    write_json(ROOT/'reports'/f'merge-{args.name}.json',report)
    print('MERGE_COMPLETE',output,flush=True)


if __name__=='__main__':main()
