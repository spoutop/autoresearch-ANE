// gpt_autoresearch.h — autoresearch GPT config for ANE training
// Matches train_mac.py GPTConfig defaults for cross-validation
// All tunable params use #ifndef to allow -D overrides for benchmarking
#pragma once

#define MODEL_NAME "GPT-autoresearch"

#ifndef DIM
#define DIM 768
#endif
#ifndef HIDDEN
#define HIDDEN 2048       // 4 * DIM (ReluSquared FFN, no gate)
#endif
#ifndef HEADS
#define HEADS 6
#endif
#ifndef KV_HEADS
#define KV_HEADS 6
#endif
#define HD (DIM/HEADS)    // = 128
#define GQA_RATIO 1       // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 768 = DIM
#define KV_DIM (KV_HEADS * HD) // = 768 = DIM
#ifndef SEQ
#define SEQ 256           // Start small for ANE testing, scale up
#endif
#ifndef NLAYERS
#define NLAYERS 8         // Default depth
#endif
#ifndef VOCAB
#define VOCAB 32768       // autoresearch vocab size
#endif

#define CKPT_PATH "ane_autoresearch_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "data/train.bin"
