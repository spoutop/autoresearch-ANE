"""ANE Karpathy Bridge — val_bpb comparison with MLX · March 8 2026
First 1:1 comparable val_bpb results from ANE on Karpathy data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re

# Colors (GitHub dark theme)
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"

# ═══════════════════════════════════════
# Parse ANE Karpathy training log
# ═══════════════════════════════════════
ane_steps, ane_bpb = [], []
ane_train_steps, ane_train_loss = [], []

with open("/Users/dan/Dev/autoresearch-ANE/results/ane_karpathy_10k.log") as f:
    for line in f:
        # Val lines: [VAL step 500] val_loss=7.4009  val_bpb=2.7556  (39677 bytes)
        m = re.search(r'\[VAL step (\d+)\].*val_bpb=(\S+)', line)
        if m:
            ane_steps.append(int(m.group(1)))
            ane_bpb.append(float(m.group(2)))
        # Training loss: step 100  loss=8.0108
        m = re.search(r'^step (\d+)\s+loss=(\S+)', line)
        if m:
            ane_train_steps.append(int(m.group(1)))
            ane_train_loss.append(float(m.group(2)))

# ═══════════════════════════════════════
# MLX best results from results.tsv
# ═══════════════════════════════════════
mlx_experiments = []
with open("/Users/dan/Dev/autoresearch-mlx/results.tsv") as f:
    next(f)  # header
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            mlx_experiments.append({
                "num": i + 1, "val_bpb": float(parts[1]),
                "status": parts[3], "desc": parts[4],
            })

mlx_best_line = []
cur = mlx_experiments[0]["val_bpb"]
for e in mlx_experiments:
    if e["status"] == "keep": cur = e["val_bpb"]
    mlx_best_line.append(cur)
mlx_best = min(e["val_bpb"] for e in mlx_experiments if e["status"] == "keep")

# ═══════════════════════════════════════
# Plot
# ═══════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.suptitle("ANE Karpathy Data Bridge — First 1:1 val_bpb Comparison",
             color=TEXT, fontsize=16, fontweight="bold", y=0.96)

# --- Left panel: ANE training curve ---
ax1 = axes[0]
ax1.set_facecolor(CARD)
ax1.plot(ane_steps, ane_bpb, 'o-', color=CYAN, linewidth=2.5, markersize=5,
         label=f'ANE val_bpb (best: {min(ane_bpb):.3f})')
ax1.axhline(y=mlx_best, color=ORANGE, linestyle='--', alpha=0.7, linewidth=1.5,
            label=f'MLX best: {mlx_best:.3f} (SEQ=1024)')
ax1.set_xlabel("Training Step", color=TEXT_DIM, fontsize=11)
ax1.set_ylabel("val_bpb (lower = better)", color=TEXT_DIM, fontsize=11)
ax1.set_title("ANE Training on Karpathy Data (10K steps)", color=TEXT, fontsize=13)
ax1.legend(loc='upper right', facecolor=CARD, edgecolor=TEXT_DIM, labelcolor=TEXT, fontsize=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.spines['bottom'].set_color(TEXT_DIM)
ax1.spines['left'].set_color(TEXT_DIM)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)

# Annotate start and end
ax1.annotate(f'{ane_bpb[0]:.3f}', (ane_steps[0], ane_bpb[0]),
             textcoords="offset points", xytext=(15, 5), color=CYAN, fontsize=9)
ax1.annotate(f'{ane_bpb[-1]:.3f}', (ane_steps[-1], ane_bpb[-1]),
             textcoords="offset points", xytext=(-45, 10), color=CYAN, fontsize=9, fontweight='bold')

# --- Right panel: Side-by-side comparison ---
ax2 = axes[1]
ax2.set_facecolor(CARD)

# Bar chart comparing the systems
systems = ['MLX\n(SEQ=1024)', 'ANE\n(SEQ=512)']
bpb_vals = [mlx_best, min(ane_bpb)]
colors = [ORANGE, CYAN]
bars = ax2.bar(systems, bpb_vals, color=colors, width=0.5, edgecolor='none', alpha=0.85)

for bar, val in zip(bars, bpb_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', color=TEXT, fontsize=14, fontweight='bold')

ax2.set_ylabel("val_bpb (lower = better)", color=TEXT_DIM, fontsize=11)
ax2.set_title("Best val_bpb — Same Data, Same Tokenizer", color=TEXT, fontsize=13)
ax2.tick_params(colors=TEXT_DIM)
ax2.spines['bottom'].set_color(TEXT_DIM)
ax2.spines['left'].set_color(TEXT_DIM)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, max(bpb_vals) * 1.15)

# Config annotations
configs = [
    f"~15M params\n1024 ctx\n59 experiments\n~69ms/step",
    f"48.8M params\n512 ctx\n1 run (10K steps)\n~139ms/step",
]
for bar, cfg in zip(bars, configs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
             cfg, ha='center', va='center', color=BG, fontsize=9, fontweight='bold')

# Subtitle with context
fig.text(0.5, 0.01,
         "Both on Karpathy climbmix-400B · rustbpe vocab=8192 · M4 Max 128GB  |  "
         "ANE gap partly from 4× shorter context (512 vs 1024) + single untuned run",
         ha='center', color=TEXT_DIM, fontsize=9)

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/ane_karpathy_comparison.png", dpi=150, facecolor=BG)
print("Saved viz/ane_karpathy_comparison.png")
print(f"\nANE best val_bpb: {min(ane_bpb):.4f} (step {ane_steps[ane_bpb.index(min(ane_bpb))]})")
print(f"MLX best val_bpb: {mlx_best:.4f}")
print(f"Gap: {min(ane_bpb) - mlx_best:.4f} bpb ({(min(ane_bpb)/mlx_best - 1)*100:.1f}% worse)")
