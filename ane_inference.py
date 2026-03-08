"""
ANE-accelerated inference using CoreML.

This script loads a CoreML model (converted from PyTorch via convert_to_coreml.py)
and runs text generation using Apple's Neural Engine where possible.

CoreML automatically dispatches operations to the most efficient compute unit:
  - ANE: Matrix multiplications, convolutions, normalization
  - GPU: Operations ANE doesn't support efficiently
  - CPU: Fallback for unsupported ops

Usage:
    python ane_inference.py --model gpt_ane.mlpackage --prompt "Once upon a time"
    python ane_inference.py --model gpt_ane.mlpackage --prompt "The meaning of" --max-tokens 200
    python ane_inference.py --model gpt_ane.mlpackage --benchmark  # performance benchmark

Requirements: macOS 15+, Apple Silicon, coremltools
"""

import os
import sys
import time
import argparse
import platform

import numpy as np

# Check we're on macOS
if sys.platform != "darwin":
    print("ERROR: ANE inference requires macOS on Apple Silicon")
    sys.exit(1)


def load_model(model_path):
    """Load a CoreML model with ANE preference."""
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Install with: pip install coremltools")
        sys.exit(1)

    print(f"Loading CoreML model: {model_path}")
    model = ct.models.MLModel(
        model_path,
        compute_units=ct.ComputeUnit.ALL,  # ANE + GPU + CPU
    )

    spec = model.get_spec()
    print(f"Model inputs: {[inp.name for inp in spec.description.input]}")
    print(f"Model outputs: {[out.name for out in spec.description.output]}")

    return model


def get_tokenizer():
    """Load the autoresearch tokenizer."""
    sys.path.insert(0, os.path.dirname(__file__))
    from prepare import Tokenizer
    return Tokenizer.from_directory()


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=50, seq_len=2048):
    """Generate text using the CoreML model (auto-dispatched to ANE/GPU/CPU)."""
    # Encode prompt
    token_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    if len(token_ids) >= seq_len:
        token_ids = token_ids[:seq_len - 1]

    generated = list(token_ids)
    print(f"\nPrompt ({len(token_ids)} tokens): {prompt}")
    print("---")
    print(prompt, end="", flush=True)

    times = []

    for i in range(max_tokens):
        # Pad or truncate to fixed sequence length (CoreML requires static shapes)
        input_ids = generated[-seq_len:]
        padding_len = seq_len - len(input_ids)
        padded = [0] * padding_len + input_ids

        # Run inference
        t0 = time.time()
        input_array = np.array([padded], dtype=np.int32)
        result = model.predict({"input_ids": input_array})
        t1 = time.time()
        times.append(t1 - t0)

        # Get logits for the last real token position
        logits = result["logits"][0, -1, :]  # (vocab_size,)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_idx = np.argsort(logits)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[top_k_idx] = logits[top_k_idx]
            logits = mask

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        generated.append(int(next_token))

        # Decode and print incrementally
        new_text = tokenizer.decode([int(next_token)])
        print(new_text, end="", flush=True)

    print("\n---")

    # Performance stats
    if times:
        avg_ms = np.mean(times) * 1000
        median_ms = np.median(times) * 1000
        tok_per_sec = 1000 / avg_ms
        print(f"Avg latency: {avg_ms:.1f}ms/token ({tok_per_sec:.1f} tok/sec)")
        print(f"Median latency: {median_ms:.1f}ms/token")
        print(f"First token: {times[0]*1000:.1f}ms (includes compilation)")
        if len(times) > 1:
            steady = times[1:]
            print(f"Steady-state: {np.mean(steady)*1000:.1f}ms/token ({1000/np.mean(steady):.1f} tok/sec)")

    return tokenizer.decode(generated)


def benchmark(model, seq_len=2048, num_runs=20, warmup=3):
    """Benchmark CoreML inference throughput."""
    print(f"\n=== ANE Inference Benchmark ===")
    print(f"Sequence length: {seq_len}")
    print(f"Warmup runs: {warmup}, Benchmark runs: {num_runs}")

    input_array = np.random.randint(0, 1000, size=(1, seq_len), dtype=np.int32)

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        model.predict({"input_ids": input_array})

    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(num_runs):
        t0 = time.time()
        model.predict({"input_ids": input_array})
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Run {i+1}/{num_runs}: {(t1-t0)*1000:.1f}ms", end="\r")

    print()
    times = np.array(times)
    print(f"\nResults ({num_runs} runs):")
    print(f"  Mean:   {times.mean()*1000:.1f}ms")
    print(f"  Median: {np.median(times)*1000:.1f}ms")
    print(f"  Std:    {times.std()*1000:.1f}ms")
    print(f"  Min:    {times.min()*1000:.1f}ms")
    print(f"  Max:    {times.max()*1000:.1f}ms")
    print(f"  Throughput: {seq_len / times.mean():.0f} tokens/sec (full sequence)")

    # Estimate ANE utilization
    # ANE typically processes at ~15 TOPS for fp16
    # A rough estimate based on known model sizes
    print(f"\nNote: Use Xcode Instruments 'Core ML Performance' template")
    print(f"to see exact ANE/GPU/CPU split for your model.")


def profile_compute_units(model_path):
    """Show which compute units CoreML plans to use for each operation."""
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed")
        sys.exit(1)

    model = ct.models.MLModel(model_path)
    spec = model.get_spec()

    # Count operation types
    if hasattr(spec, 'mlProgram'):
        print("\n=== CoreML Model Analysis ===")
        print("This model uses the ML Program format (good for ANE).")
        print("\nANE-friendly operations (typically dispatched to ANE):")
        print("  - Linear/MatMul layers")
        print("  - Layer/RMS normalization")
        print("  - Softmax")
        print("  - Element-wise ops (add, mul, relu)")
        print("\nOperations that may fall back to GPU/CPU:")
        print("  - Rotary embeddings (complex indexing)")
        print("  - Attention masking (dynamic)")
        print("  - Tanh softcap")
        print("\nUse 'Instruments > Core ML Performance' for actual dispatch info.")
    else:
        print("Model uses Neural Network format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANE-accelerated GPT inference")
    parser.add_argument("--model", default="gpt_ane.mlpackage", help="CoreML model path")
    parser.add_argument("--prompt", default=None, help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seq-len", type=int, default=2048, help="Model sequence length")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of generation")
    parser.add_argument("--profile", action="store_true", help="Profile compute unit usage")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print(f"First train a model and convert it:")
        print(f"  python train_mac.py")
        print(f"  python convert_to_coreml.py")
        sys.exit(1)

    if args.profile:
        profile_compute_units(args.model)
    elif args.benchmark:
        model = load_model(args.model)
        benchmark(model, seq_len=args.seq_len)
    else:
        model = load_model(args.model)
        tokenizer = get_tokenizer()
        prompt = args.prompt or "Once upon a time in a land far away"
        generate(model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seq_len=args.seq_len)
