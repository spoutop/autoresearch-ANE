# Experiment Results — autoresearch-ANE

## Hardware
- Apple M4 Max, 128GB unified memory
- 40 GPU cores (MPS), 16 ANE cores, ~32MB ANE SRAM

## Directory Structure

### `mps_pretest/`
MPS/Metal GPU pre-test sweep results from autoresearch-macos.
- Depth sweep: D={2,4,6,8} at batch 16, total 65K
- Batch sweep: B={16,32,64,128,256} at depth 4
- Winner: **Depth 4, Batch 32, val_bpb = 1.309**
- Key finding: More steps > bigger batches on Metal (opposite of GPU)

### `ane_bench_*.tsv` + `logs_*/`
ANE hardware profiling — 15 configs (NL={2,4,8,16,24} × SEQ={256,512,1024}).
- All 15 compile and train successfully
- SRAM wall at SEQ=1024 (SEQ=1152+ fails)
- 6-8% ANE utilization (600-800 GFLOP/s of 10.5 TFLOP/s peak)
- Timing breakdown: 33% ANE compute, 30% IO, 37% CPU

### `sweep_5min/`
ANE 5-minute training sweep with real climbmix data.
- 5 candidate configs, each trained for ~5 minutes
- First real loss numbers on ANE with real data
- Answers: which config is compute-optimal for ANE?

## Key Numbers

| System | Best val_bpb | Steps/5min | Tokens/run | Config |
|--------|-------------|------------|------------|--------|
| H100 (Karpathy) | 0.998 | 953 | 499.6M | D8, B128, S2048 |
| M4 Max MPS | 1.309 | 393 | 25.8M | D4, B32, S2048 |
| M4 Max ANE | TBD | TBD | TBD | TBD |
