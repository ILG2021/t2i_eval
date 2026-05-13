import torch
from safetensors.torch import load_file
from .model import ErnieImageModel

def load_ernie_fp8(ckpt_path, device="cpu"):
    raw_sd = load_file(ckpt_path, device=device)
    
    # Filter and remap keys
    unet_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model.diffusion_model."):
            unet_sd[k[len("model.diffusion_model."):]] = v
        else:
            unet_sd[k] = v
            
    # Process Forge/ComfyUI style FP8 weights
    processed_sd = {}
    for k, v in unet_sd.items():
        if v.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            scale_key = k.replace(".weight", ".weight_scale")
            if scale_key in unet_sd:
                scale = unet_sd[scale_key]
                processed_sd[k] = (v.to(torch.float32) * scale).to(torch.bfloat16)
            else:
                processed_sd[k] = v.to(torch.bfloat16)
        elif ".weight_scale" in k or ".scale_weight" in k:
            continue
        else:
            processed_sd[k] = v
            
    # Model configuration
    config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_layers": 36,
        "ffn_hidden_size": 12288,
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "text_in_dim": 3072,
        "rope_theta": 256,
        "rope_axes_dim": (32, 48, 48),
    }
    
    model = ErnieImageModel(**config)
    model.load_state_dict(processed_sd, strict=False)
    return model
