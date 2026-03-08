"""
Convert a trained GPT checkpoint to CoreML format for ANE inference.

The Apple Neural Engine (ANE) accelerates CoreML models automatically.
This script converts the PyTorch GPT model to CoreML, with optimizations
for ANE execution:
  - Float16 precision (ANE's native format)
  - Static shapes (ANE requires fixed tensor shapes)
  - Channel-first layout hints where possible

Usage:
    python convert_to_coreml.py                           # default checkpoint
    python convert_to_coreml.py --checkpoint my_model.pt  # custom checkpoint
    python convert_to_coreml.py --seq-len 512             # shorter sequence for faster inference

Requirements: pip install coremltools (macOS only)
"""

import os
import sys
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_head < self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPTForExport(nn.Module):
    """GPT model simplified for CoreML export (no window_size dynamics)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.vocab_size, config.n_embd, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().half(), freqs.sin().half()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)
        return logits


def convert(checkpoint_path, output_path, seq_len=None):
    """Convert PyTorch checkpoint to CoreML model."""
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Install with:")
        print("  pip install coremltools")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]

    if seq_len:
        config_dict["sequence_len"] = seq_len

    config = GPTConfig(**config_dict)
    print(f"Model config: {config}")

    # Build export model and load weights
    model = GPTForExport(config)

    # Load state dict — handle possible key mismatches from compiled model
    state_dict = ckpt["model_state_dict"]
    # Strip torch.compile prefixes if present
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model.half()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Trace with example input
    T = config.sequence_len
    example_input = torch.randint(0, config.vocab_size, (1, T))

    print(f"Tracing model with input shape (1, {T})...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, T), dtype=int),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,  # Let CoreML decide CPU/GPU/ANE split
    )

    # Set model metadata
    mlmodel.author = "autoresearch"
    mlmodel.short_description = f"GPT-{sum(p.numel() for p in model.parameters()) // 1_000_000}M for text generation"
    mlmodel.version = "1.0"

    # Save
    mlmodel.save(output_path)
    model_size_mb = os.path.getsize(output_path) / 1024 / 1024 if os.path.isdir(output_path) else 0
    print(f"\nCoreML model saved to: {output_path}")
    print(f"Model will use ANE automatically where possible.")
    print(f"\nTo profile ANE usage, use Xcode Instruments > Core ML Performance")
    print(f"Or run: python ane_inference.py --model {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GPT checkpoint to CoreML for ANE inference")
    parser.add_argument("--checkpoint", default="checkpoint_mac.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="gpt_ane.mlpackage", help="Output CoreML model path")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length (shorter = faster)")
    args = parser.parse_args()

    convert(args.checkpoint, args.output, args.seq_len)
