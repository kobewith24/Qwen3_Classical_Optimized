"""Explicit frozen IO precision; keep norms and trainable LoRA in FP32."""
TARGET_MODULES=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def configure_frozen_io(model,precision='fp32'):
    import torch
    if precision not in {'fp32','bf16'}:raise ValueError('Unsupported frozen IO precision')
    modules={'input_embeddings':model.get_input_embeddings(),'output_head':model.get_output_embeddings()}
    changed=set();result={}
    for name,module in modules.items():
        if module is None:continue
        if any(p.requires_grad for p in module.parameters()):raise ValueError('IO precision requires frozen weights')
        if any(p.__class__.__name__=='Params4bit' for p in module.parameters()):raise ValueError('Do not recast quantized IO weights')
        if precision=='bf16' and id(module) not in changed:module.to(dtype=torch.bfloat16);changed.add(id(module))
        result[name]={'dtype':str(module.weight.dtype),'parameters':sum(p.numel() for p in module.parameters()),'trainable':False}
    if precision=='bf16' and torch.cuda.is_available():torch.cuda.empty_cache()
    return result
