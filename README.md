# autoresearch-ANE

**Apple Silicon LLM training — three accelerators, one chip.**

Autonomous AI research on M4 Max using all three compute paths Apple Silicon offers: the Apple Neural Engine (ANE) via native Obj-C, the GPU via MLX, and the GPU via PyTorch/MPS. Forked from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Same protocol: an AI agent modifies training code, runs 5-minute experiments, evaluates `val_bpb`, keeps or discards, and loops overnight. But instead of one H100, we're running on a laptop chip — and discovering what works (and what doesn't) on Apple Silicon.

## Results so far

**ANE (native Obj-C, Apple Neural Engine):**
- [NEW] Fixed native framework pipeline: implemented programmatic in-memory MIL/weight generation to circumvent missing CoreML package dependencies (`make test-ane` executing at `0.11 TFLOPS`)
- 67.6M param GPT, 6 layers, SEQ=512, ~99ms/step
- Best loss: 5.81 (LR=2e-4, 10K steps)
- ANE is invisible to Activity Monitor — runs alongside GPU with zero interference
- Key challenge: activation instability on long runs (cosine schedule must match run length)

**MPS (PyTorch, Metal GPU):**
- [NEW] Enabled and stabilized PyTorch MPS backend for ~64GB unified memory (M4 Pro)
- [NEW] **OOM Fix:** Reduced default `DEVICE_BATCH_SIZE` and increased accumulation to fit smaller memory pools
- [NEW] **NaN / Overflow Fix:** Disabled PyTorch `autocast` (falling back to stable `float32`) and converted hardcoded `float16` embedding casts to `bfloat16` natively to prevent massive gradient overflows
- [NEW] **Optimizer Fix:** Corrected CPU/MPS tensor device mismatches in the Muon optimizer `lerp_` operations
- 11.5M param GPT, val_bpb=1.308 after 79 autonomous experiments
- bf16 confirmed 2.6x slower on Apple Silicon — fp32 is faster
- H100 findings (embedding WD, init scaling) do not transfer to MPS

**MLX (Apple's native ML framework) — [`mlx/`](mlx/):**
- [NEW] Fully enabled and verified pipeline on M4 Pro (`gpu:0` compute, ~22k tok/sec out-of-the-box)
- ~50M param GPT, val_bpb=1.665 baseline (agent optimizing now)
- Native bf16, unified memory, no translation layer
- Ported from [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## Quick start

### ANE (native, macOS Apple Silicon only)

```bash
cd native && make all
make test-ane              # verify ANE hardware access
make bench-sram            # probe SRAM performance cliffs
./build/train_overnight_nl6_s512 --steps 10000 --scratch --lr 2e-4 \
  --data data/train.bin --val data/val.bin
```

### MLX (recommended for Apple Silicon GPU)

```bash
cd mlx && uv sync
uv run prepare.py --num-shards 8
uv run train.py
```

### MPS (PyTorch / Apple Silicon)

```bash
cp pyproject_mac.toml pyproject.toml && uv sync
uv run prepare.py --num-shards 8
uv run train_mac.py
```

### Autonomous agent mode

```bash
claude --dangerously-skip-permissions -p "Read program.md and start autoresearch."
```

## Architecture

```
native/             — ANE hardware-level training (Obj-C, private APIs)
  runtime/          — ANE interface (_ANEInMemoryModel, IOSurface)
  mil/              — MIL code generation, dynamic weight pipeline
  training/         — training loop, CPU fallback ops (RMSNorm, Adam)
  bridge/           — C API for Python ctypes
  probes/           — hardware exploration (SRAM limits, weight patching)

mlx/                — MLX GPU training (Apple's native ML framework)
  train.py          — model + optimizer + loop (agent modifies this)
  prepare.py        — data prep, tokenizer, evaluation (read-only)
  program.md        — agent instructions

train.py            — NVIDIA GPU training (upstream, CUDA)
train_mac.py        — Apple Silicon training (MPS backend)
prepare.py          — data prep, tokenizer, evaluation (read-only)
program.md          — agent instructions
viz/                — result visualizations
```

### Key concept: dynamic weight pipeline (ANE)

Weights are packed into the IOSurface input alongside activations. Kernels compile once at startup; weight updates are just `memcpy` — no recompilation needed. This is the core innovation over [maderix/ANE](https://github.com/maderix/ANE) which rebaked weights into compiled kernels.

## Key findings

- **ANE: 6x bigger model, 8x faster** than MPS on the same chip
- **Both accelerators run simultaneously** with zero interference
- **ANE timing breakdown:** 33% ANE compute, 30% IO, 37% CPU (classifier is 22% bottleneck)
- **Depth U-curve at SEQ=512:** NL=4(6.74) → NL=6(6.34) → NL=8(6.94) → NL=12(7.14)
- **SRAM wall at SEQ=1024** — ANE runs out of on-chip memory
- **Cosine schedule length must match actual run length** or activations explode

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch concept and nanochat
- [trevin-creator](https://github.com/trevin-creator) — [MLX port](https://github.com/trevin-creator/autoresearch-mlx) that this repo's `mlx/` is based on
- [miolini](https://github.com/miolini) — [MPS/macOS port](https://github.com/miolini/autoresearch-macos)
- [maderix](https://github.com/maderix) — [ANE private API](https://github.com/maderix/ANE) reverse engineering
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
