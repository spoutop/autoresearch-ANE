# Pre-test Results — M4 Max (128GB) MPS/Metal

## Hardware
- Apple M4 Max, 128GB unified RAM
- macOS, Metal/MPS backend
- No torch.compile, no FlashAttention 3, no bf16 autocast

## Depth Sweep (batch 16, total 65K)

| Depth | Params | Steps | val_bpb |
|-------|--------|-------|---------|
| 2 | 3.5M | 1,002 | 1.391 |
| **4** | **11.5M** | **368** | **1.312** |
| 6 | 26.3M | 198 | 1.418 |
| 8 | 50.3M | 107 | 1.576 |

Winner: Depth 4. GPU default (depth 8) is 20% worse.

## Batch Sweep (depth 4)

| Device Batch | Total Batch | Steps | val_bpb | Notes |
|-------------|-------------|-------|---------|-------|
| 16 | 65K | 368 | 1.312 | Fork default |
| **32** | **65K** | **393** | **1.309** | **BEST** |
| 64 | 128K | 205 | 1.403 | |
| 128 | 256K | 89 | 1.672 | |
| 256 | 512K | - | OOM | 141GB used |
| 32 | 256K | 81 | 1.706 | Bigger total hurts |
| 32 | 512K | 48 | 1.832 | GPU-level total = terrible |

Winner: Batch 32, Total 65K. More steps > bigger batches on Metal.

## Optimal MPS Config

```
DEPTH = 4
DEVICE_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 2**16  (65K)
WINDOW_PATTERN = "L"
ASPECT_RATIO = 64
```

val_bpb = 1.309, 393 steps, 11.5M params

## Key Findings

1. On Metal, more training steps beats bigger batches — opposite of GPU conventional wisdom
2. GPU default config (depth 8, batch 128) scores 1.576 on Mac — 20% worse than optimized
3. Batch 256 OOMs at 141GB on 128GB machine
4. Fork defaults (depth 4, batch 16) were close but not optimal

## Still To Test
- bf16 autocast on MPS (potential 2x speed)
- Depth 3 and 5 (fine-tuning around sweet spot)
- Aspect ratio tuning
- Overnight autonomous agent run

## Comparison to GPU (Karpathy's H100)

| Metric | H100 | M4 Max (MPS) |
|--------|------|-------------|
| Baseline val_bpb | 0.998 | 1.312 |
| Steps / 5 min | 953 | 393 |
| Tokens / run | 499.6M | 25.8M |
| Model params | 50.3M | 11.5M |
| Depth | 8 | 4 |
| Batch size | 128 | 32 |
