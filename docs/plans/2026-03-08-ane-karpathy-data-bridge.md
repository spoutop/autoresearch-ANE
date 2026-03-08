# ANE Karpathy Data Bridge — Implementation Plan

> **Goal:** Run ANE native training on the same Karpathy climbmix-400B data, same tokenizer, and same val_bpb evaluation as MLX/MPS — enabling 1:1 comparison across all three accelerators.

**Architecture:** Python script converts Karpathy data → flat binary files that native Obj-C code reads. Evaluation runs in Python by loading ANE-trained weights and calling `evaluate_bpb()`.

---

## What needs to change

The ANE native code (Obj-C) currently uses:
- Custom pre-tokenized `train.bin` / `val.bin`
- Unknown tokenizer and vocab size
- SEQ=512
- Raw training loss as the metric

To compare 1:1, we need:
- **Same data**: Karpathy climbmix-400B (parquet → tokenized → binary)
- **Same tokenizer**: rustbpe, vocab=8192 (matching MLX)
- **Same eval**: `evaluate_bpb()` from MLX's `prepare.py`
- **SEQ=512 is OK** — ANE hits SRAM wall at 1024, so we keep 512 but acknowledge it's shorter than MLX's 2048

## Tasks

### Task 1: Data conversion script

**Files:**
- Create: `native/scripts/convert_karpathy_data.py`

**What it does:**
1. Load Karpathy's rustbpe tokenizer from `~/.cache/autoresearch/tokenizer/`
2. Read parquet shards from `~/.cache/autoresearch/data/`
3. Tokenize all documents with BOS prepending
4. Pack into SEQ=512+1 rows (simple sequential, no best-fit packing needed for binary)
5. Write flat `int32` binary files: `native/data/train_karpathy.bin`, `native/data/val_karpathy.bin`
6. Also write `native/data/token_bytes.bin` (int32 array, byte count per token — needed for bpb eval)

**Format:** Just a flat array of int32 tokens. The native code reads `SEQ+1` tokens at a time (input + target). This matches how `train.m` already reads `train.bin`.

**Key detail:** vocab size must be 8192 to match the rustbpe tokenizer. The ANE model's embedding table and classifier need to be sized for vocab=8192.

```bash
cd native && python scripts/convert_karpathy_data.py
# Output: data/train_karpathy.bin, data/val_karpathy.bin, data/token_bytes.bin
```

### Task 2: Update ANE model config for vocab=8192

**Files:**
- Modify: `native/training/models/gpt_autoresearch.h`
- Or create: `native/training/models/gpt_karpathy.h`

**Changes:**
- Set `VOCAB_SIZE = 8192` (currently different)
- Keep everything else: NLAYERS=6, DIM=768, SEQ=512, NHEADS, etc.

```bash
# Build with new config
cd native && make MODEL=gpt_karpathy train
```

### Task 3: Add val_bpb evaluation to native training

**Option A (simpler): Python eval wrapper**

Create `native/scripts/eval_ane_bpb.py` that:
1. Loads the ANE checkpoint (`ane_autoresearch_dyn_ckpt.bin`)
2. Wraps it in a Python model that calls the ANE bridge for forward pass
3. Calls `evaluate_bpb()` from MLX's `prepare.py`

This is complex because the ANE bridge would need to support inference-only forward pass from Python.

**Option B (recommended): Native bpb calculation in train.m**

Add to `train.m`:
1. Load `token_bytes.bin` at startup
2. During validation, compute per-token cross-entropy (already done)
3. Multiply each token's CE by whether `token_bytes[token_id] > 0`
4. Sum nats, sum bytes, compute `nats / (ln(2) * total_bytes)` = bpb

This is straightforward — just a few lines in the validation loop.

**Changes to train.m:**
```c
// Load token_bytes
int32_t *token_bytes = load_binary("data/token_bytes.bin");

// In validation loop, after computing per-token loss:
float total_nats = 0;
int total_bytes = 0;
for (int i = 0; i < seq_len; i++) {
    int target_id = targets[i];
    int nbytes = token_bytes[target_id];
    if (nbytes > 0) {
        total_nats += per_token_loss[i];
        total_bytes += nbytes;
    }
}
float val_bpb = total_nats / (logf(2.0f) * total_bytes);
```

### Task 4: Run ANE with Karpathy data

```bash
cd native
./build/train_karpathy_nl6_s512 --steps 10000 --scratch --lr 2e-4 --clip 1.0 \
  --data data/train_karpathy.bin --val data/val_karpathy.bin \
  --val-interval 2000 --val-steps 10
```

The output will now include `val_bpb` — directly comparable to MLX and MPS.

### Task 5: Fair comparison run

Run all three with matched conditions:

| Setting | ANE | MLX | MPS |
|---------|-----|-----|-----|
| Data | climbmix-400B | climbmix-400B | climbmix-400B |
| Tokenizer | rustbpe 8192 | rustbpe 8192 | tiktoken 32768* |
| SEQ | 512 | 2048 | 2048 |
| Eval | val_bpb | val_bpb | val_bpb |
| Time budget | 5 min | 5 min | 5 min |

*MPS uses different tokenizer — for true 1:1, MPS would also need rustbpe. But MPS is retired, so ANE vs MLX is the priority.

**Note on SEQ=512 vs 2048:** ANE can't do 2048 (SRAM wall). This means ANE sees 4x less context per sample. The val_bpb will be higher (worse) than MLX partly for this reason. To be fully fair, we could also run MLX at SEQ=512 as a control.

## Order of operations

1. **Task 1** — Convert data (30 min, Python script)
2. **Task 2** — Update model config (10 min, header file)
3. **Task 3** — Add bpb eval to train.m (30 min, C code)
4. **Task 4** — Test run (20 min)
5. **Task 5** — Fair comparison + visualization (30 min)

Total: ~2 hours of work, then we have a true 1:1 comparison.
