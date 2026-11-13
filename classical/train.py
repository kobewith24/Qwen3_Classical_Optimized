"""QLoRA with explicit verified assistant labels and proportional warmup."""
import argparse
import importlib.metadata
import json
import math
import os
import re
import time
from collections import Counter

from .core import ROOT, digest, file_digest, read_jsonl, write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--name', default='clean_2k_seed24')
    ap.add_argument('--data-dir',default='data/prepared',help='Versioned prepared data inside this project')
    ap.add_argument('--validate-only',action='store_true',help='Verify frozen files and labels without loading a GPU model')
    ap.add_argument('--frozen-io-dtype',choices=['fp32','bf16'],default='fp32')
    ap.add_argument('--seed', type=int, default=24); ap.add_argument('--epochs', type=float, default=1)
    ap.add_argument('--rank', type=int, default=32); ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1.5e-4); args=ap.parse_args()
    # Optional versioned data is added after both recorded control runs; their exact
    # training source remains in reports/source-snapshot/classical/train.py.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.name):raise ValueError('Invalid experiment name')
    if args.samples<1 or args.epochs<=0 or args.rank<1 or args.lr<=0:raise ValueError('Training settings must be positive')
    data=(ROOT/args.data_dir).resolve();output=(ROOT/'weights'/args.name).resolve();checkpoint=(ROOT/'checkpoints'/args.name).resolve()
    if not all(p.is_relative_to(ROOT.resolve()) for p in (data,output,checkpoint)):
        raise ValueError('Training paths must stay inside the new project')
    if output.exists() or checkpoint.exists() or (ROOT/'reports'/f'train-{args.name}.json').exists():
        raise FileExistsError('Experiment already exists; choose a new --name to preserve recorded weights and results')
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=(data/name).resolve()
        if not path.is_relative_to(data) or file_digest(path)!=expected:
            raise ValueError('Prepared file hash mismatch: '+name)
    available=list(read_jsonl(data/'train-tokenized.jsonl'))
    if args.samples>len(available):raise ValueError('Requested samples exceed the frozen dataset')
    raw=available[:args.samples];dev=list(read_jsonl(data/'dev-tokenized.jsonl'))
    if not raw or not dev:raise ValueError('Training and development data must be nonempty')
    for row in raw+dev:
        assert len(row['input_ids'])==len(row['labels'])==len(row['attention_mask'])<=1024
        assert row['labels'][:row['prompt_tokens']]==[-100]*row['prompt_tokens']
        assert row['labels'][row['prompt_tokens']:]==row['input_ids'][row['prompt_tokens']:]
    if args.validate_only:
        print(json.dumps({'validated':True,'model_loaded':False,'name':args.name,'samples':len(raw),'dev_samples':len(dev),
            'data_dir':str(data.relative_to(ROOT)),'manifest_sha256':file_digest(data/'manifest.json'),
            'effective_supervised_tokens':sum(r['supervised_tokens'] for r in raw)},ensure_ascii=False));return
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments,set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from .training_utils import TARGET_MODULES,configure_frozen_io
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    # Seed before constructing the LoRA A matrices, not only inside Trainer.
    set_seed(args.seed)
    initialization_rng_sha256=digest(torch.get_rng_state().numpy().tobytes())
    training_source_sha256=file_digest(__file__)
    manifest_sha256=file_digest(data/'manifest.json')
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model', local_files_only=True)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    dataset=lambda rows:Dataset.from_list([{k:r[k] for k in ['input_ids','attention_mask','labels']} for r in rows])
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'assets/base_model',local_files_only=True,quantization_config=quant,device_map={'':0},attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    io_precision=configure_frozen_io(model,args.frozen_io_dtype)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=0,bias='none',task_type='CAUSAL_LM',
        target_modules=TARGET_MODULES))
    model.print_trainable_parameters()
    trainer=Trainer(model=model,processing_class=tokenizer,train_dataset=dataset(raw),eval_dataset=dataset(dev),
        data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,pad_to_multiple_of=8,label_pad_token_id=-100),
        args=TrainingArguments(output_dir=str(checkpoint),per_device_train_batch_size=1,per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,num_train_epochs=args.epochs,learning_rate=args.lr,
            warmup_steps=max(1,math.ceil(math.ceil(len(raw)/8)*args.epochs*.05)),
            lr_scheduler_type='cosine',optim='paged_adamw_8bit',bf16=True,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='steps',eval_steps=50,
            save_strategy='steps',save_steps=50,save_total_limit=1,load_best_model_at_end=True,
            metric_for_best_model='eval_loss',greater_is_better=False,logging_steps=10,report_to=[],
            dataloader_num_workers=0,seed=args.seed,data_seed=args.seed,disable_tqdm=True))
    initialization_peak={'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    torch.cuda.reset_peak_memory_stats()
    started=time.time(); result=trainer.train()
    training_peak={'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    output.mkdir(parents=True,exist_ok=True); model.save_pretrained(output); tokenizer.save_pretrained(output)
    stats={'name':args.name,'seed':args.seed,'epochs':args.epochs,'samples':len(raw),'tasks':dict(Counter(r['task'] for r in raw)),
        'effective_supervised_tokens':sum(r['supervised_tokens'] for r in raw),'warmup_ratio':.05,'lr':args.lr,'lora_rank':args.rank,
        'runtime_seconds':time.time()-started,'gpu':torch.cuda.get_device_name(),
        'peak_allocated_gib':max(initialization_peak['allocated_gib'],training_peak['allocated_gib']),
        'peak_reserved_gib':max(initialization_peak['reserved_gib'],training_peak['reserved_gib']),
        'initialization_peak_gib':initialization_peak,'training_peak_gib':training_peak,
        'frozen_io_precision':args.frozen_io_dtype,'frozen_io_modules':io_precision,
        'memory_scope':'CUDA allocator counters; initialization included, not an independent WDDM physical/shared residency measurement',
        'support_code_sha256':file_digest(ROOT/'classical/training_utils.py'),
        'train_metrics':result.metrics,'best_checkpoint':trainer.state.best_model_checkpoint,'history':trainer.state.log_history,
        'dataset_manifest_sha256':manifest_sha256,'initialization_seed_controlled':True,
        'data_dir':str(data.relative_to(ROOT)),'prepared_file_hashes_verified':True,
        'initialization_rng_sha256':initialization_rng_sha256,'training_source_sha256':training_source_sha256,
        'label_policy':'explicit prompt labels=-100, assistant and EOS supervised; no truncation',
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','datasets','accelerate']}}
    stats['adapter_sha256']=file_digest(output/'adapter_model.safetensors')
    assert file_digest(data/'manifest.json')==manifest_sha256,'Frozen dataset changed during training'
    write_json(ROOT/'reports'/f'train-{args.name}.json',stats)
    print('TRAINING_COMPLETE',output,flush=True)


if __name__=='__main__':main()
