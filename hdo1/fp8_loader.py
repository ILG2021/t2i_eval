from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor


FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

SAFETENSORS_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}
SAFETENSORS_DTYPE_MAP = {key: value for key, value in SAFETENSORS_DTYPE_MAP.items() if value is not None}


def _is_float8_dtype(dtype: torch.dtype | None) -> bool:
    return dtype in FLOAT8_DTYPES


def _dtype_from_safetensors(model_dir: Path) -> torch.dtype | None:
    candidates = []
    single = model_dir / "model.safetensors"
    if single.exists():
        candidates.append(single)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for shard_name in sorted(set(index.get("weight_map", {}).values())):
                shard_path = model_dir / shard_name
                if shard_path.exists():
                    candidates.append(shard_path)
        except Exception:
            pass

    counts: dict[torch.dtype, int] = {}
    for path in candidates[:2]:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    dtype_name = handle.get_slice(key).get_dtype()
                    dtype = SAFETENSORS_DTYPE_MAP.get(dtype_name)
                    if dtype is not None:
                        counts[dtype] = counts.get(dtype, 0) + 1
        except Exception:
            continue

    for dtype in FLOAT8_DTYPES:
        if counts.get(dtype, 0) > 0:
            return dtype
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        if counts.get(dtype, 0) > 0:
            return dtype
    return None


def _compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _add_special_tokens(tokenizer) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


class FP8Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_dtype = getattr(self, "hidream_compute_dtype", None)
        if compute_dtype is None or not _is_float8_dtype(self.weight.dtype):
            return super().forward(input)
        weight = self.weight.to(dtype=compute_dtype)
        bias = self.bias.to(dtype=compute_dtype) if self.bias is not None and _is_float8_dtype(self.bias.dtype) else self.bias
        return F.linear(input.to(dtype=compute_dtype), weight, bias)


class FP8Conv3d(nn.Conv3d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_dtype = getattr(self, "hidream_compute_dtype", None)
        if compute_dtype is None or not _is_float8_dtype(self.weight.dtype):
            return super().forward(input)
        weight = self.weight.to(dtype=compute_dtype)
        bias = self.bias.to(dtype=compute_dtype) if self.bias is not None and _is_float8_dtype(self.bias.dtype) else self.bias
        return F.conv3d(input.to(dtype=compute_dtype), weight, bias, self.stride, self.padding, self.dilation, self.groups)


class FP8Embedding(nn.Embedding):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_dtype = getattr(self, "hidream_compute_dtype", None)
        if compute_dtype is None or not _is_float8_dtype(self.weight.dtype):
            return super().forward(input)
        return F.embedding(
            input,
            self.weight.to(dtype=compute_dtype),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def _linear_wrapper(module: nn.Linear, compute_dtype: torch.dtype) -> FP8Linear:
    wrapped = FP8Linear(module.in_features, module.out_features, module.bias is not None, device="meta", dtype=module.weight.dtype)
    wrapped.weight = module.weight
    wrapped.bias = module.bias
    wrapped.hidream_compute_dtype = compute_dtype
    return wrapped


def _conv3d_wrapper(module: nn.Conv3d, compute_dtype: torch.dtype) -> FP8Conv3d:
    wrapped = FP8Conv3d(
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
        module.bias is not None,
        module.padding_mode,
        device="meta",
        dtype=module.weight.dtype,
    )
    wrapped.weight = module.weight
    wrapped.bias = module.bias
    wrapped.hidream_compute_dtype = compute_dtype
    return wrapped


def _embedding_wrapper(module: nn.Embedding, compute_dtype: torch.dtype) -> FP8Embedding:
    wrapped = FP8Embedding(
        module.num_embeddings,
        module.embedding_dim,
        module.padding_idx,
        module.max_norm,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
        device="meta",
        dtype=module.weight.dtype,
    )
    wrapped.weight = module.weight
    wrapped.hidream_compute_dtype = compute_dtype
    return wrapped


def _wrap_fp8_modules(module: nn.Module, compute_dtype: torch.dtype) -> None:
    for name, child in list(module.named_children()):
        _wrap_fp8_modules(child, compute_dtype)
        wrapped = None
        if isinstance(child, nn.Linear) and not isinstance(child, FP8Linear):
            wrapped = _linear_wrapper(child, compute_dtype)
        elif isinstance(child, nn.Conv3d) and not isinstance(child, FP8Conv3d):
            wrapped = _conv3d_wrapper(child, compute_dtype)
        elif isinstance(child, nn.Embedding) and not isinstance(child, FP8Embedding):
            wrapped = _embedding_wrapper(child, compute_dtype)
        if wrapped is not None:
            setattr(module, name, wrapped)
    module.hidream_compute_dtype = compute_dtype


def _set_parameter(root: nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], nn.Parameter(value, requires_grad=False))


def _get_module(root: nn.Module, dotted_path: str) -> nn.Module | None:
    module = root
    for part in dotted_path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def _recast_dtype_probe_modules(model: nn.Module, compute_dtype: torch.dtype) -> int:
    recast = 0
    for module_path in ("model.visual.patch_embed.proj", "model.visual.pos_embed", "model.t_embedder1.mlp.0"):
        module = _get_module(model, module_path)
        if module is None:
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if _is_float8_dtype(param.dtype):
                setattr(module, name, nn.Parameter(param.detach().to(dtype=compute_dtype), requires_grad=False))
                recast += 1
    return recast


def _recast_small_fp8_tensors(model: nn.Module, compute_dtype: torch.dtype) -> int:
    recast = 0
    for name, param in list(model.named_parameters()):
        if not _is_float8_dtype(param.dtype):
            continue
        if param.ndim >= 2 and not name.endswith(".bias"):
            continue
        _set_parameter(model, name, param.detach().to(dtype=compute_dtype))
        recast += 1
    return recast


def _restore_rotary_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if "RotaryEmbedding" not in module.__class__.__name__:
            continue
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is None or getattr(inv_freq, "is_meta", False):
            if hasattr(module, "rope_init_fn") and hasattr(module, "config"):
                inv_freq, attention_scaling = module.rope_init_fn(module.config, device=torch.device("cpu"))
                module.attention_scaling = attention_scaling
            else:
                size = tuple(getattr(inv_freq, "shape", (0,)))
                dim = int(size[0] * 2) if size and size[0] else 0
                if dim <= 0:
                    continue
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq


def _load_single_safetensors(model: nn.Module, model_dir: Path) -> None:
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Expected FP8 weights at {weights_path}")
    state_dict = load_file(str(weights_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    missing = [key for key in missing if not key.endswith(".inv_freq")]
    if unexpected:
        print(f"[fp8] Ignored {len(unexpected)} unexpected weight keys.")
    if missing:
        print(f"[fp8] Missing {len(missing)} model keys after direct safetensors load.")


def load_image_model(model_path: str | Path):
    model_dir = Path(model_path).expanduser().resolve()
    print(f"[fp8] Loading checkpoint from {model_dir} ...")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    _add_special_tokens(_get_tokenizer(processor))

    weight_dtype = _dtype_from_safetensors(model_dir)
    if not _is_float8_dtype(weight_dtype):
        raise RuntimeError(f"Expected FP8 safetensors in {model_dir}, detected {weight_dtype}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HiDream O1 FP8 inference.")

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    compute_dtype = _compute_dtype()
    print(f"[fp8] Detected {weight_dtype} weights; using {compute_dtype} compute.")
    config = Qwen3VLForConditionalGeneration.config_class.from_pretrained(str(model_dir))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    _load_single_safetensors(model, model_dir)
    _restore_rotary_buffers(model)
    dtype_probe_recast = _recast_dtype_probe_modules(model, compute_dtype)
    _wrap_fp8_modules(model, compute_dtype)
    recast = _recast_small_fp8_tensors(model, compute_dtype)
    if dtype_probe_recast:
        print(f"[fp8] Recast {dtype_probe_recast} dtype-probe tensors to {compute_dtype}.")
    if recast:
        print(f"[fp8] Recast {recast} small FP8 tensors to {compute_dtype}.")
    model.hidream_dtype = compute_dtype
    model.hidream_weight_dtype = weight_dtype
    model.to("cuda").eval()
    gc.collect()
    torch.cuda.empty_cache()
    return processor, model
