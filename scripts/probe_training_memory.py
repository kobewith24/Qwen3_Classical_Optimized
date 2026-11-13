"""Paired same-batch backward probes, not a replacement for full training."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,digest,file_digest,read_jsonl,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--precision',choices=['fp32','bf16'],required=True);args=ap.parse_args()
    output=ROOT/'reports/2026-11-13'/f'memory-probe-{args.precision}.json'
    if output.exists():raise FileExistsError(output)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from classical.training_utils import TARGET_MODULES,configure_frozen_io
    set_seed(24);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    rows=list(read_jsonl(ROOT/'data/prepared/train-tokenized.jsonl'))
    items=[max((r for r in rows if r['task']==task),key=lambda r:len(r['input_ids'])) for task in ('punctuate','c2m','m2c')]
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model',local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'assets/base_model',local_files_only=True,quantization_config=quant,device_map={'':0},attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    io=configure_frozen_io(model,args.precision);model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=TARGET_MODULES))
    model.train();initial={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();losses=[]
    for row in items:
        batch={key:torch.tensor([row[key]],device='cuda') for key in ('input_ids','attention_mask','labels')}
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
        losses.append(float(loss.detach()));(loss/len(items)).backward()
    torch.cuda.synchronize()
    name,param=next((n,p) for n,p in model.named_parameters() if 'layers.0.self_attn.q_proj.lora_B' in n)
    gradient=param.grad.detach().float().cpu().numpy()
    if not np.isfinite(gradient).all() or not np.linalg.norm(gradient):raise ValueError('Invalid adapter gradient')
    gradient_path=output.with_suffix('.gradient.npy');np.save(gradient_path,gradient)
    norm_dtypes=sorted({str(p.dtype) for n,p in model.named_parameters() if 'norm' in n})
    write_json(output,{'precision':args.precision,'seed':24,'samples':[r['sample_id'] for r in items],
        'lengths':[len(r['input_ids']) for r in items],'losses':losses,'backward_seconds':time.perf_counter()-started,
        'initialization_peak_gib':initial,'backward_peak_gib':{'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30},
        'io':io,'norm_dtypes':norm_dtypes,'gradient_parameter':name,'gradient_sha256':file_digest(gradient_path),
        'training_data_manifest_sha256':file_digest(ROOT/'data/prepared/manifest.json'),
        'scope':'Three longest task examples, BF16 autocast, backward only; no optimizer step or physical/shared residency guarantee'})
    print('MEMORY_PROBE_COMPLETE',args.precision,flush=True)

if __name__=='__main__':main()
