import contextlib
import threading
import time

from .core import ROOT, digest, file_digest, messages


class HFEngine:
    """One resident NF4 base; adapters switch serially, never concurrently."""
    def __init__(self, batch_size=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the local NF4 backend')
        torch.set_num_threads(4)
        self.torch=torch; self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/base_model',local_files_only=True)
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.model=AutoModelForCausalLM.from_pretrained(ROOT/'assets/base_model',local_files_only=True,
            quantization_config=q,device_map={'':0},attn_implementation='sdpa')
        self.model.eval(); self.loaded=set(); self.current='base'; self.adapters={}
        self.template_hash=digest(self.tokenizer.chat_template)

    def select(self, name, path=None):
        if name=='base':self.current=name;return
        if name not in self.loaded:
            from peft import PeftModel
            if not self.loaded:self.model=PeftModel.from_pretrained(self.model,path,adapter_name=name,local_files_only=True)
            else:self.model.load_adapter(path,adapter_name=name,local_files_only=True)
            self.loaded.add(name)
            self.adapters[name]=file_digest(path/'adapter_model.safetensors')
        self.model.set_adapter(name); self.model.eval(); self.current=name

    def fingerprint(self):
        import json
        import importlib.metadata
        manifest=ROOT/'reports/assets.json'
        if not manifest.exists():raise RuntimeError('Run scripts/verify_assets.py before model evaluation')
        base=json.loads(manifest.read_text(encoding='utf-8'))['base_model_fingerprint']
        return digest({'base':base,'adapter':self.adapters.get(self.current),'template':self.template_hash,
            'quantization':'nf4-double-bf16','backend':'transformers','batch_size':self.batch_size,
            'embedding_dtype':str(self.model.get_input_embeddings().weight.dtype),
            'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes']}})

    def generate_messages(self, turns, max_new_tokens=256, cancel=None, timeout=120, streamer=None):
        return self.generate_prompts([turns],max_new_tokens=max_new_tokens,cancel=cancel,timeout=timeout,streamer=streamer)[0]

    def generate_prompts(self, turns_list, max_new_tokens=256, cancel=None, timeout=120, streamer=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        cancel=cancel or threading.Event(); deadline=time.monotonic()+timeout
        class Stop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):return cancel.is_set() or time.monotonic()>=deadline
        prompts=[self.tokenizer.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=False) for turns in turns_list]
        enc=self.tokenizer(prompts,return_tensors='pt',padding=True,truncation=False)
        if enc['input_ids'].shape[1]>1024:raise ValueError('Input exceeds 1024 tokens; split the source text')
        enc=enc.to(self.model.device); start=time.perf_counter(); self.torch.cuda.synchronize()
        mode=self.model.disable_adapter() if self.current=='base' and self.loaded else contextlib.nullcontext()
        extra={'streamer':streamer} if streamer is not None else {}
        with mode,self.torch.inference_mode():
            generated=self.model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,
                repetition_penalty=1.,pad_token_id=self.tokenizer.pad_token_id,use_cache=True,
                stopping_criteria=StoppingCriteriaList([Stop()]),**extra)
        self.torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        if cancel.is_set():raise InterruptedError('Generation cancelled')
        if time.monotonic()>=deadline:raise TimeoutError('Generation exceeded total deadline')
        width=enc['input_ids'].shape[1]; eos=self.model.generation_config.eos_token_id
        endings=set(eos if isinstance(eos,list) else [eos]); output=[]
        for i,sequence in enumerate(generated[:,width:]):
            ids=sequence.tolist(); count=len(ids)
            for n,token in enumerate(ids):
                if token in endings:count=n+1;break
            clipped=ids[:count]
            output.append({'text':self.tokenizer.decode(clipped,skip_special_tokens=True).strip(),
                'completion_tokens':count,'prompt_tokens':int(enc['attention_mask'][i].sum()),
                'batch_seconds':elapsed,'finish_reason':'length' if count==max_new_tokens and clipped[-1] not in endings else 'stop'})
        return output

    def generate_items(self, items, max_new_tokens=320):
        return self.generate_prompts([messages(item) for item in items],max_new_tokens=max_new_tokens)
