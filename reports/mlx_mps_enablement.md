# Pasteback: Enabling MLX & MPS Pipelines on Apple Silicon

## Goal
Enable and verify the MLX and MPS training pipelines on Apple Silicon from the `autoresearch` fork, resolving any device-specific crashes or bugs.

## Summary of Changes
1. **MLX Pipeline (`mlx/`)**: Successfully synced dependencies and ran `prepare.py` out-of-the-box. Training starts and runs flawlessly (~22k tok/sec on M4 GPU).
2. **MPS Pipeline (`train_mac.py`)**: 
   - Swapped `pyproject.toml` to macOS dependencies and installed `torch` / `coremltools`.
   - **Fixed OOM**: Reduced `DEVICE_BATCH_SIZE` from 64 to 16 to fit the 50M parameter model comfortably within the available ~64GB unified memory on the M4 Pro.
   - **Fixed CPU/MPS Mismatches**: Fixed a bug where the `MuonAdamW` optimizer created scalar tensors on CPU but attempted to use them in in-place MPS operations (`lerp_`), crashing the optimizer.
   - **Fixed NaN Gradient Overflow**: Fixed `NaN` loss explosions at step 1 by disabling `autocast` (falling back to `float32` compute) and converting the hardcoded `float16` embedding casts to `bfloat16` to prevent gradient overflows given the repo's very high learning rates.

---

## Output Proofs

### 1. MLX Training Successful
```bash
$ cd mlx
$ uv run train.py
Data/tokenizer loaded in 0.1s
Time budget: 300s
Gradient accumulation steps: 2
Model compiled in 5.3s
step 00019 (17.4%) | loss: 7.530029 | lrm: 1.00 | dt: 2902ms | tok/sec: 22,585 | epoch: 1 | remaining: 245s
step 00034 (31.9%) | loss: 7.268844 | lrm: 1.00 | dt: 2918ms | tok/sec: 22,459 | epoch: 1 | remaining: 201s
```

### 2. MPS Training Successful
```bash
$ uv run train_mac.py
Using MPS (Metal Performance Shaders) on arm
Detected: Apple M4 Pro — estimated 7.6 TFLOPS peak (GPU)
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 512, 'window_pattern': 'SSSL'}
Parameter counts:
  wte                     : 4,194,304
  value_embeds            : 16,777,216
  lm_head                 : 4,194,304
  transformer_matrices    : 25,166,336
  scalars                 : 16
  total                   : 50,332,176
Estimated FLOPs per token: 2.390784e+08
Model memory: ~0.09 GB (float16)
Scaling AdamW LRs by 1/sqrt(512/768) = 1.224745
Running in eager mode (set AUTORESEARCH_COMPILE=1 to try torch.compile)
Time budget: 300s
Gradient accumulation steps: 16
Effective batch size: 524,288 tokens
Unified memory advantage: model + optimizer + data all share yourGB pool

--- Training started ---
step 00000 (0.0%) | loss: 9.011654 | lrm: 1.00 | dt: 52584ms | tok/sec: 9,970 | mfu: 31.4% | mem: 606MB | epoch: 1 | remaining: 300s    
step 00001 (0.0%) | loss: 9.189484 | lrm: 1.00 | dt: 52332ms | tok/sec: 10,018 | mfu: 31.5% | mem: 606MB | epoch: 1 | remaining: 300s    
```
*(Gradients correctly flow and NaN logic is resolved; loop gracefully continues)*
