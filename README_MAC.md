# autoresearch-ANE: Mac Silicon + Apple Neural Engine Edition

Autonomous LLM pretraining research on Apple Silicon, adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with Apple Neural Engine (ANE) integration inspired by [maderix/ANE](https://github.com/maderix/ANE).

## What's Different from Upstream

| Feature | Original (CUDA) | This Fork (Mac Silicon) |
|---------|-----------------|------------------------|
| GPU Backend | CUDA (H100) | MPS (Metal Performance Shaders) |
| Attention | Flash Attention 3 | PyTorch native SDPA |
| Precision | bfloat16 | float16 (MPS native) |
| Compilation | torch.compile | Eager mode (optional compile) |
| Inference | GPU | ANE via CoreML |
| Memory | GPU VRAM (80GB H100) | Unified Memory (up to 192GB) |

## Quick Start

```bash
# 1. Set up dependencies (on your Mac)
cp pyproject_mac.toml pyproject.toml
pip install uv
uv sync

# 2. Prepare data (downloads ~1GB of training data)
uv run prepare.py --num-shards 8

# 3. Train on Mac Silicon (MPS backend)
uv run train_mac.py

# 4. Convert to CoreML for ANE inference
uv run convert_to_coreml.py

# 5. Run inference on ANE
uv run ane_inference.py --prompt "Once upon a time"
```

## Architecture

### Training: MPS Backend (`train_mac.py`)

Training uses PyTorch's MPS (Metal Performance Shaders) backend, which runs on the Apple Silicon GPU. Key adaptations:

- **Flash Attention 3 → native SDPA**: PyTorch's `scaled_dot_product_attention` works on MPS and auto-selects the best kernel
- **bfloat16 → float16**: MPS has full float16 support; bfloat16 is limited
- **No torch.compile**: MPS compilation support is evolving; eager mode is reliable
- **Unified memory**: Your 128GB is shared between CPU and GPU — no separate VRAM limit!

### Inference: ANE via CoreML (`ane_inference.py`)

For inference, we convert the trained model to CoreML format. CoreML automatically dispatches operations to the most efficient compute unit:

- **ANE**: Matrix multiplications, normalizations, activations (~70% of ops)
- **GPU**: Complex operations ANE doesn't support
- **CPU**: Fallback for remaining ops

### Direct ANE Access (`ane_bridge.py`)

Experimental module inspired by [maderix/ANE](https://github.com/maderix/ANE) that provides:

- ANE hardware detection and capability reporting
- MIL (Model Intermediate Language) program generation
- Performance estimation and backend comparison
- Reference for building native ANE training loops

## Compute Backends Compared

### For Training

| Backend | TFLOPS | Maturity | Ease of Use |
|---------|--------|----------|-------------|
| **MPS (GPU)** | 7-30 (chip dependent) | Stable | High — standard PyTorch |
| **ANE (direct)** | 11-76 TOPS (5-9% util.) | Experimental | Low — Obj-C, private APIs |
| **CPU** | ~1-2 | Stable | High — always works |

**Recommendation**: Use MPS for training. ANE direct training (maderix/ANE approach) achieves only 5-9% of peak throughput due to CPU-ANE data transfer overhead.

### For Inference

| Backend | Throughput | Power Efficiency | Ease of Use |
|---------|-----------|-----------------|-------------|
| **CoreML/ANE** | Good | Excellent | Medium — needs conversion |
| **MPS (GPU)** | Good | Good | High — use model directly |
| **CPU** | Fair | Fair | High — always works |

**Recommendation**: Use CoreML/ANE for inference deployment. It's power-efficient and handles dispatch automatically.

## Model Size Guide (128GB Unified Memory)

With 128GB, you can train much larger models than typical GPU setups:

| DEPTH | Params | Model Memory | Optimizer Memory | Total ~Est. | Status |
|-------|--------|-------------|-----------------|-------------|--------|
| 8 | 50M | 0.1 GB | 0.3 GB | ~1 GB | Default, trains fast |
| 16 | 200M | 0.4 GB | 1.2 GB | ~3 GB | Good sweet spot |
| 24 | 450M | 0.9 GB | 2.7 GB | ~6 GB | Comfortable |
| 32 | 800M | 1.6 GB | 4.8 GB | ~10 GB | Still easy |
| 48 | 1.8B | 3.6 GB | 10.8 GB | ~22 GB | Ambitious |
| 64 | 3.2B | 6.4 GB | 19.2 GB | ~40 GB | Feasible on 128GB! |

To change model size, edit `DEPTH` in `train_mac.py`. The model dimension scales as `depth * ASPECT_RATIO`.

## ANE Deep Dive

### How the ANE Works (from maderix/ANE research)

The Apple Neural Engine is a dedicated hardware accelerator in Apple Silicon:

- **Interface**: Private `_ANEClient` API, not publicly documented
- **Programs**: Compiled from MIL (Model Intermediate Language) text
- **Data format**: IOSurface shared memory, `[1, C, 1, S]` layout, float16
- **Per-step kernels** (for a transformer layer):
  1. `kFwdAttn` — RMSNorm + QKV + SDPA + output projection
  2. `kFwdFFN` — RMSNorm + SwiGLU
  3. `kFFNBwd` — FFN backward
  4. `kSdpaBwd1/2` — Attention backward (2 parts)
  5. `kQKVb` — QKV backward
- **CPU offload**: RMSNorm backward, residuals, loss, weight gradients (cblas), Adam

### Current ANE Limitations

- ~5-9% of peak TOPS utilization during training
- No native causal masking in ANE SDPA
- ~119 compile limit per process (resource leak)
- Element-wise ops fall back to CPU
- Private API — may break across macOS versions

### Future Possibilities

- Apple may expose training-friendly ANE APIs in future macOS/Xcode releases
- CoreML improvements could enable training via the public API
- MLX (Apple's ML framework) may add ANE dispatch
- Community work on maderix/ANE may improve utilization

## Files

| File | Purpose |
|------|---------|
| `train_mac.py` | Training script for Mac Silicon (MPS backend) |
| `convert_to_coreml.py` | Convert trained model to CoreML for ANE |
| `ane_inference.py` | Run inference using CoreML/ANE |
| `ane_bridge.py` | Experimental direct ANE access + hardware info |
| `pyproject_mac.toml` | macOS-compatible dependencies |
| `prepare.py` | Data preparation (device-agnostic, works on Mac) |
| `train.py` | Original CUDA training script (upstream reference) |

## Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the brilliant autonomous research framework
- [maderix/ANE](https://github.com/maderix/ANE) — pioneering reverse-engineering of Apple Neural Engine training
- The PyTorch MPS backend team at Apple
