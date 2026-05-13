"""
ERNIE-Image FP8 Loader for t2i_eval

Replicates Forge's loading pipeline:
  load_torch_file (safe_open) → convert_quantization → strip prefix → load into model

Key insight: Abiray/ERNIE-Image-Turbo-FP8-NVFP4 uses ComfyUI's "scaled_fp8" format:
  - One tensor ending in "scaled_fp8" stores the global dtype sentinel
  - Per-layer scale tensors use ".scale_weight" suffix (NOT ".weight_scale")
  - Forge renames: .scale_weight → .weight_scale, .scale_input → .input_scale
  - We dequantize FP8 * scale_weight → bfloat16 for diffusers compatibility
"""

import torch

from .model import ErnieImageModel

ERNIE_CONFIG = {
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


def _load_safetensors(ckpt_path: str) -> tuple[dict, dict]:
    """
    Manual safetensors reader that bypasses the strict 'file not fully covered'
    validation added in safetensors >= 0.4.0.

    ComfyUI's scaled_fp8 format files often have extra bytes at the end that
    aren't declared in the header (padding/alignment), which causes the standard
    safe_open to raise: "incomplete metadata, file not fully covered".

    safetensors binary layout:
      [8 bytes: header_size (uint64 LE)]
      [header_size bytes: JSON header]
      [remaining bytes: raw tensor data]

    Each tensor entry in the JSON has:
      {"dtype": ..., "shape": [...], "data_offsets": [start, end]}
    where offsets are relative to the start of the data section.
    """
    import json
    import struct

    DTYPE_MAP = {
        "F64":  torch.float64,
        "F32":  torch.float32,
        "BF16": torch.bfloat16,
        "F16":  torch.float16,
        "F8_E4M3": getattr(torch, "float8_e4m3fn",  torch.float32),
        "F8_E5M2": getattr(torch, "float8_e5m2",    torch.float32),
        "I64":  torch.int64,
        "I32":  torch.int32,
        "I16":  torch.int16,
        "I8":   torch.int8,
        "U8":   torch.uint8,
        "BOOL": torch.bool,
    }

    with open(ckpt_path, "rb") as f:
        # 1. Read 8-byte header size
        header_size = struct.unpack("<Q", f.read(8))[0]
        # 2. Read JSON header
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))
        # 3. Data section starts here
        data_start = 8 + header_size
        # 4. Read full data section into a buffer (bypass coverage check)
        data = f.read()  # reads everything remaining

    metadata = header.pop("__metadata__", {}) or {}

    sd = {}
    for name, info in header.items():
        dtype_str = info["dtype"]
        shape     = info["shape"]
        start, end = info["data_offsets"]

        dtype = DTYPE_MAP.get(dtype_str)
        if dtype is None:
            print(f"[ERNIE Loader] Unknown dtype '{dtype_str}' for {name}, skipping")
            continue

        raw = data[start:end]
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
        if shape:
            tensor = tensor.reshape(shape)
        else:
            tensor = tensor.squeeze()
        sd[name] = tensor.clone()  # clone to detach from buffer

    return sd, metadata


def _convert_quantization(sd: dict) -> dict:
    """
    Replicates Forge's convert_quantization from backend/state_dict.py.
    Handles ComfyUI scaled_fp8 format:
      - finds the 'scaled_fp8' sentinel key to detect FP8 dtype
      - renames .scale_weight → .weight_scale
      - renames .scale_input → .input_scale (drops if value == 1.0)
    """
    # Find the sentinel key (e.g. "model.diffusion_model.scaled_fp8")
    model_prefix = None
    for key in sd.keys():
        if key.endswith("scaled_fp8"):
            model_prefix = key.replace("scaled_fp8", "")
            break

    if model_prefix is None:
        # Not a scaled_fp8 file – return as-is
        return sd

    scaled_fp8_key = f"{model_prefix}scaled_fp8"
    scaled_fp8_weight = sd[scaled_fp8_key]
    scaled_fp8_dtype = scaled_fp8_weight.dtype
    if scaled_fp8_dtype is torch.float32:
        scaled_fp8_dtype = torch.float8_e4m3fn

    print(f"[ERNIE Loader] Detected scaled_fp8 format, dtype={scaled_fp8_dtype}")

    out_sd = {}
    for k in list(sd.keys()):
        if k == scaled_fp8_key:
            continue
        if not k.startswith(model_prefix):
            out_sd[k] = sd[k]
            continue

        k_out = k
        w = sd[k]

        if k_out.endswith(".scale_weight"):
            # rename to .weight_scale (matching Forge convention)
            layer = k_out[: -len(".scale_weight")]
            k_out = f"{layer}.weight_scale"
        elif k_out.endswith(".scale_input"):
            # rename to .input_scale, drop if trivially 1.0
            layer = k_out[: -len(".scale_input")]
            k_out = f"{layer}.input_scale"
            if w.item() == 1.0:
                continue

        out_sd[k_out] = w

    return out_sd


def _strip_prefix(sd: dict) -> dict:
    """Remove 'model.diffusion_model.' prefix (Forge's preprocess_state_dict)."""
    PREFIX = "model.diffusion_model."
    if not any(k.startswith(PREFIX) for k in sd.keys()):
        return sd
    return {k[len(PREFIX):] if k.startswith(PREFIX) else k: v for k, v in sd.items()}


def _dequantize_fp8(sd: dict) -> dict:
    """
    Dequantize FP8 weights for use with standard diffusers linear layers.
    For each FP8 weight tensor: dequant = weight.float() * weight_scale → bfloat16
    """
    fp8_types = set()
    for t in [getattr(torch, f"float8_{s}", None) for s in ("e4m3fn", "e4m3fnuz", "e5m2", "e5m2fnuz")]:
        if t is not None:
            fp8_types.add(t)

    out = {}
    scale_keys = {k for k in sd if k.endswith(".weight_scale") or k.endswith(".input_scale")}

    for k, v in sd.items():
        if k in scale_keys:
            continue  # skip scale tensors after processing
        if v.dtype in fp8_types:
            scale_key = k.replace(".weight", ".weight_scale")
            if scale_key in sd:
                scale = sd[scale_key].float()
                out[k] = (v.float() * scale).to(torch.bfloat16)
                print(f"[ERNIE Loader] Dequantized {k}: FP8 * scale → bfloat16")
            else:
                out[k] = v.to(torch.bfloat16)
        else:
            out[k] = v

    return out


def load_ernie_fp8(ckpt_path: str) -> ErnieImageModel:
    """
    Load an ERNIE-Image FP8 safetensors checkpoint into an ErnieImageModel.

    Steps:
      1. safe_open (avoids mmap issues on remote filesystems)
      2. convert_quantization (handle scaled_fp8 sentinel + rename scale keys)
      3. strip model.diffusion_model. prefix
      4. dequantize FP8 → bfloat16 (diffusers compatibility)
      5. load_state_dict into ErnieImageModel
    """
    print(f"[ERNIE Loader] Loading checkpoint: {ckpt_path}")

    # Step 1: Load via safe_open
    sd, metadata = _load_safetensors(ckpt_path)
    print(f"[ERNIE Loader] Loaded {len(sd)} tensors")

    # Step 2: Handle scaled_fp8 quantization format
    sd = _convert_quantization(sd)

    # Step 3: Strip model.diffusion_model. prefix
    sd = _strip_prefix(sd)

    # Step 4: Dequantize FP8 → bfloat16
    sd = _dequantize_fp8(sd)

    # Step 5: Initialize model and load weights
    model = ErnieImageModel(**ERNIE_CONFIG)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"[ERNIE Loader] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ERNIE Loader] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model
