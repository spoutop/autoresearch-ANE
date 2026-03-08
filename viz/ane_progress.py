"""ANE Training Progress — From First Boot to Stable 30K Steps"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import re

# Colors
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
RED = "#f85149"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"
GOLD = "#e3b341"

def parse_log(path):
    steps, losses, xmaxs = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('step '):
                parts = line.split()
                try:
                    step = int(parts[1])
                    loss = float(parts[2].replace('loss=', ''))
                    m = re.search(r'x\[([^,]+),([^\]]+)\]', line)
                    if m:
                        xmax = max(abs(float(m.group(1))), abs(float(m.group(2))))
                    else:
                        xmax = 0
                    steps.append(step)
                    losses.append(loss)
                    xmaxs.append(xmax)
                except (ValueError, IndexError):
                    pass
    return np.array(steps), np.array(losses), np.array(xmaxs)

def smooth(arr, window=30):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# Load all runs
runs = {}

# Overnight v1 (DIVERGED)
s, l, x = parse_log("/Users/dan/Dev/autoresearch-ANE/results/overnight_ane_nl6_s512.log")
if len(s) > 0:
    runs["v1: LR=3e-4, 330K steps"] = {"s": s, "l": l, "x": x, "color": RED, "status": "DIVERGED"}

# Overnight v2 (DIVERGED)
s, l, x = parse_log("/Users/dan/Dev/autoresearch-ANE/results/overnight_ane_nl6_s512_v2.log")
if len(s) > 0:
    runs["v2: LR=1e-4, 330K steps"] = {"s": s, "l": l, "x": x, "color": ORANGE, "status": "DIVERGED"}

# Test E (STABLE)
s, l, x = parse_log("/tmp/test_e.log")
if len(s) > 0:
    runs["E: LR=2e-4, 10K steps"] = {"s": s, "l": l, "x": x, "color": GREEN, "status": "STABLE"}

# Test F (STABLE — best!)
s, l, x = parse_log("/tmp/test_f.log")
if len(s) > 0:
    runs["F: LR=1e-4, 30K steps"] = {"s": s, "l": l, "x": x, "color": CYAN, "status": "STABLE"}

# Test G (STABLE)
s, l, x = parse_log("/tmp/test_g.log")
if len(s) > 0:
    runs["G: LR=3e-4, 5K steps"] = {"s": s, "l": l, "x": x, "color": PURPLE, "status": "STABLE"}

# ═══════════════════════════════════════
# Figure
# ═══════════════════════════════════════
fig = plt.figure(figsize=(24, 16), facecolor=BG)

fig.text(0.5, 0.97, "ANE Training — From Explosion to Stability", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.943, "67.6M param GPT on Apple Neural Engine · M4 Max · Finding the right config",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 3, left=0.06, right=0.97, top=0.90, bottom=0.08,
                       hspace=0.35, wspace=0.3)

# ═══════════════════════════════════════
# Panel 1: Loss curves — all runs (top left, 2 cols)
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")

for name, r in runs.items():
    l_smooth = smooth(r["l"], window=40)
    s_plot = r["s"][:len(l_smooth)]
    lw = 3 if "F:" in name else 2
    alpha = 0.9 if r["status"] == "STABLE" else 0.5
    ax1.plot(s_plot, l_smooth, color=r["color"], linewidth=lw, alpha=alpha, label=name)

# Mark divergence
for name, r in runs.items():
    if r["status"] == "DIVERGED":
        diverge_idx = np.where(r["x"] > 50)[0]
        if len(diverge_idx) > 0:
            div_step = r["s"][diverge_idx[0]]
            ax1.axvline(x=div_step, color=r["color"], linewidth=1, linestyle=":", alpha=0.4)

ax1.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("Training Loss", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("Loss Curves — Overnight Failures vs Stable Tests", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
ax1.legend(loc="upper right", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)
ax1.set_ylim(4.5, 9.5)

# ═══════════════════════════════════════
# Panel 2: Progress timeline (top right)
# ═══════════════════════════════════════
ax_card = fig.add_subplot(gs[0, 2], facecolor=CARD)
for spine in ax_card.spines.values(): spine.set_color("#30363d")
ax_card.set_xlim(0, 10)
ax_card.set_ylim(0, 10)
ax_card.set_xticks([])
ax_card.set_yticks([])

ax_card.text(5, 9.3, "PROGRESS TIMELINE", fontsize=15, fontweight="bold",
             color=GOLD, ha="center", fontfamily="monospace")
ax_card.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

timeline = [
    ("DAY 1", "First ANE training ever", TEXT_DIM, ""),
    ("", "Proved 67.6M model trains at 99ms/step", GREEN, ""),
    ("", "", "", ""),
    ("NIGHT 1", "Overnight v1: LR=3e-4", RED, "DIVERGED at 15K"),
    ("NIGHT 2", "Overnight v2: LR=1e-4", RED, "DIVERGED at 15K"),
    ("", "Root cause: cosine schedule mismatch", GOLD, ""),
    ("", "", "", ""),
    ("DAY 2", "Stability test sweep (5 configs)", CYAN, ""),
    ("", "Test E: LR=2e-4, 10K — STABLE", GREEN, "loss 5.72"),
    ("", "Test F: LR=1e-4, 30K — STABLE", GREEN, "loss 5.03 (BEST)"),
    ("", "Test G: LR=3e-4, 5K — STABLE", GREEN, "loss 5.53"),
    ("", "", "", ""),
    ("NOW", "Ready for overnight v3", GOLD, "proven config"),
]

y = 8.2
for phase, desc, color, note in timeline:
    if desc == "":
        y -= 0.25
        continue
    if phase:
        ax_card.text(0.5, y, phase, fontsize=9, fontweight="bold",
                     color=GOLD if phase == "NOW" else TEXT_DIM,
                     fontfamily="monospace")
    ax_card.text(2.8, y, desc, fontsize=8, color=color, fontfamily="monospace")
    if note:
        ax_card.text(9.0, y, note, fontsize=7, color=color, fontfamily="monospace",
                     ha="right", alpha=0.7)
    y -= 0.55

# ═══════════════════════════════════════
# Panel 3: Activation stability (bottom left)
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")

for name, r in runs.items():
    x_smooth = smooth(r["x"], window=20)
    s_plot = r["s"][:len(x_smooth)]
    lw = 2.5 if r["status"] == "STABLE" else 1.5
    alpha = 0.9 if r["status"] == "STABLE" else 0.5
    ax2.plot(s_plot, x_smooth, color=r["color"], linewidth=lw, alpha=alpha, label=name)

ax2.axhline(y=50, color=RED, linewidth=1.5, linestyle="--", alpha=0.5)
ax2.text(200, 70, "EXPLODES (>50)", fontsize=8, color=RED, fontfamily="monospace", alpha=0.7)
ax2.axhline(y=5, color=GOLD, linewidth=1, linestyle="--", alpha=0.3)
ax2.text(200, 6.5, "WARNING (>5)", fontsize=7, color=GOLD, fontfamily="monospace", alpha=0.5)

ax2.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("|x| max activation", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("Activation Magnitude — The Stability Signal", fontsize=13,
              color=TEXT, fontfamily="monospace", pad=10)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.set_yscale("log")
ax2.legend(loc="upper left", fontsize=7, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 4: Best loss comparison (bottom center)
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")

configs = ["v1\n330K", "v2\n330K", "G\n5K", "E\n10K", "F\n30K"]
best_losses = [
    min(runs["v1: LR=3e-4, 330K steps"]["l"]) if "v1: LR=3e-4, 330K steps" in runs else 9,
    min(runs["v2: LR=1e-4, 330K steps"]["l"]) if "v2: LR=1e-4, 330K steps" in runs else 9,
    min(runs["G: LR=3e-4, 5K steps"]["l"]) if "G: LR=3e-4, 5K steps" in runs else 9,
    min(runs["E: LR=2e-4, 10K steps"]["l"]) if "E: LR=2e-4, 10K steps" in runs else 9,
    min(runs["F: LR=1e-4, 30K steps"]["l"]) if "F: LR=1e-4, 30K steps" in runs else 9,
]
bar_colors = [RED, ORANGE, PURPLE, GREEN, CYAN]
statuses = ["DIVERGED", "DIVERGED", "STABLE", "STABLE", "STABLE (BEST)"]

x_pos = np.arange(len(configs))
bars = ax3.bar(x_pos, best_losses, color=bar_colors, alpha=0.8, width=0.6,
               edgecolor="#30363d", linewidth=1)

for i, (bar, loss, status) in enumerate(zip(bars, best_losses, statuses)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{loss:.2f}", fontsize=11, color=TEXT, fontfamily="monospace",
             ha="center", fontweight="bold")
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
             status, fontsize=7, color=bar_colors[i], fontfamily="monospace",
             ha="center", fontweight="bold", alpha=0.8)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(configs, fontsize=9, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Best Training Loss", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_title("Best Loss by Config — Lower is Better", fontsize=13,
              color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")
ax3.set_ylim(4.5, 6.5)

# Arrow showing improvement
ax3.annotate("", xy=(4, best_losses[4] - 0.05), xytext=(0, best_losses[0] + 0.1),
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5, connectionstyle="arc3,rad=-0.3"))
ax3.text(2, 6.2, "progress", fontsize=11, color=GREEN, fontfamily="monospace",
         fontweight="bold", ha="center", rotation=-15)

# ═══════════════════════════════════════
# Panel 5: Key numbers (bottom right)
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2], facecolor=CARD)
for spine in ax4.spines.values(): spine.set_color("#30363d")
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.3, "BY THE NUMBERS", fontsize=15, fontweight="bold",
         color=GOLD, ha="center", fontfamily="monospace")
ax4.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

numbers = [
    ("67.6M", "parameters", TEXT),
    ("6 layers", "DIM=768, SEQ=512", TEXT_DIM),
    ("99ms", "per step (peak, ANE)", GREEN),
    ("", "", ""),
    ("5.03", "best loss (Test F, 30K steps)", CYAN),
    ("5.72", "best loss (Test E, 10K steps)", GREEN),
    ("5.86", "best loss (v1, before diverge)", RED),
    ("", "", ""),
    ("2", "overnight runs failed (diverged)", RED),
    ("3", "stability tests passed", GREEN),
    ("1", "root cause found (schedule mismatch)", GOLD),
    ("", "", ""),
    ("NEXT", "overnight v3 with proven config", GOLD),
]

y = 8.1
for val, label, color in numbers:
    if val == "":
        y -= 0.3
        continue
    ax4.text(1.0, y, val, fontsize=13, fontweight="bold",
             color=color, ha="left", fontfamily="monospace")
    ax4.text(4.5, y, label, fontsize=9, color=TEXT_DIM,
             ha="left", fontfamily="monospace", va="center")
    y -= 0.65

# Center watermark
fig.text(0.5, 0.50, "@danpacary", fontsize=60, color=TEXT_DIM, alpha=0.04,
         ha="center", va="center", fontfamily="monospace", fontweight="bold",
         rotation=25, zorder=0)

# Bottom
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Apple Neural Engine · M4 Max",
         fontsize=8, color=TEXT_DIM, alpha=0.3, ha="left", va="bottom", fontfamily="monospace")

out = "/Users/dan/Dev/autoresearch-ANE/viz/ane_progress.png"
plt.savefig(out, dpi=180, facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
