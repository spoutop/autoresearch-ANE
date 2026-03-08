#!/bin/bash
# bench_ane.sh — Systematic ANE training benchmark suite
# Compiles and runs all config combinations, logs results to TSV
#
# Usage: ./bench_ane.sh [steps]    (default: 30 steps per config)

set -euo pipefail
cd "$(dirname "$0")/native"

STEPS=${1:-30}
RESULTS_DIR="../results"
mkdir -p "$RESULTS_DIR" build data

# Create dummy data if needed
if [ ! -f data/train.bin ]; then
    echo "Creating dummy training data (1M tokens, valid IDs 0-8191)..."
    python3 -c "
import struct, random
random.seed(42)
tokens = [random.randint(0, 8191) for _ in range(1_000_000)]
with open('data/train.bin', 'wb') as f:
    f.write(struct.pack(f'<{len(tokens)}H', *tokens))
"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$RESULTS_DIR/ane_bench_${TIMESTAMP}.tsv"
LOGDIR="$RESULTS_DIR/logs_${TIMESTAMP}"
mkdir -p "$LOGDIR"

CC="xcrun clang"
BASE_FLAGS="-O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK"
INCLUDES="-I. -Iruntime -Imil -Itraining -include training/models/gpt_autoresearch.h"
LDFLAGS="-ldl -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate"

echo "=== ANE Training Benchmark ==="
echo "Steps per config: $STEPS"
echo "Results: $OUTFILE"
echo ""

# TSV header
printf "nlayers\tseq\tparams_M\tflops_B\tms_step\ttokens_step\tcompile_ms\tstatus\n" > "$OUTFILE"

for NL in 2 4 8 16 24; do
    for S in 256 512 1024; do
        BIN="build/train_nl${NL}_s${S}"
        LOG="$LOGDIR/nl${NL}_s${S}.log"

        echo -n "NL=$NL SEQ=$S: "

        # Compile
        $CC $BASE_FLAGS -DNLAYERS=$NL -DSEQ=$S $INCLUDES -o "$BIN" training/train.m $LDFLAGS 2>/dev/null

        # Run
        if ! OUTPUT=$("./$BIN" --steps "$STEPS" --scratch 2>&1); then
            # Check if it was a compile failure at runtime (ANE compiler)
            if echo "$OUTPUT" | grep -q "Compilation failed"; then
                echo "SRAM FAIL (kernel compile)"
                printf "%d\t%d\t-\t-\t-\t%d\t-\tSRAM_FAIL\n" "$NL" "$S" "$S" >> "$OUTFILE"
                echo "$OUTPUT" > "$LOG"
                continue
            fi
            echo "RUNTIME FAIL"
            printf "%d\t%d\t-\t-\t-\t%d\t-\tRUNTIME_FAIL\n" "$NL" "$S" "$S" >> "$OUTFILE"
            echo "$OUTPUT" > "$LOG"
            continue
        fi

        # Save full log
        echo "$OUTPUT" > "$LOG"

        # Parse results
        PARAMS=$(echo "$OUTPUT" | grep "Params:" | sed 's/Params: \([0-9.]*\)M.*/\1/')
        FLOPS=$(echo "$OUTPUT" | grep "FLOPs/step:" | sed 's/.*total=\([0-9.]*\)M/\1/' | awk '{printf "%.1f", $1/1000}')
        MS_STEP=$(echo "$OUTPUT" | grep "Train time:" | sed 's/.*(\(.*\)ms\/step)/\1/')
        COMPILE=$(echo "$OUTPUT" | grep "Compiled 10" | sed 's/.*in \([0-9]*\)ms.*/\1/')

        echo "${MS_STEP}ms/step (${PARAMS}M params, ${FLOPS}B FLOPs)"
        printf "%d\t%d\t%s\t%s\t%s\t%d\t%s\tOK\n" "$NL" "$S" "$PARAMS" "$FLOPS" "$MS_STEP" "$S" "$COMPILE" >> "$OUTFILE"
    done
done

echo ""
echo "=== Results ==="
column -t -s$'\t' "$OUTFILE"
echo ""
echo "Full logs: $LOGDIR/"
echo "TSV data:  $OUTFILE"
