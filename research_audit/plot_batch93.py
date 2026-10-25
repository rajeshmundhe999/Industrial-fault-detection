"""
Batch 93 Score Curve Investigation
====================================
Plots the GRU-AE v2 anomaly score for batch 93 over all windows
to determine whether the trend alarm at window 11 is a genuine
early warning or a spurious false alarm.

Also plots all 10 faulty batches (91-100) for comparison.

Usage:
  py -3.12 research_audit/plot_batch93.py

Output:
  research_audit/figures/batch93_score_curve.png
  research_audit/figures/all_faulty_batches_scores.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "model", "results", "leakage_safe_v2")
EARLY = os.path.join(BASE, "model", "results", "early_detection")
FIG_DIR = os.path.join(BASE, "research_audit", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────────────
d = np.load(os.path.join(RESULTS, "scores_and_predictions.npz"), allow_pickle=True)
scores = d["end_robust_mean_val_max_score"]   # primary operating point
y = d["y_end"]
batch_ids = d["batch_ids"]
end_times = d["end_times"]

op = np.load(os.path.join(RESULTS, "operating_point.npz"), allow_pickle=True)
threshold = float(op["threshold"])
print(f"Static threshold: {threshold:.4f}")

# Load composite scores (trend-based early detection)
composite_path = os.path.join(EARLY, "composite_vs_static_auc.csv")
import csv
composite_data = {}
if os.path.exists(composite_path):
    with open(composite_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch = int(row.get("batch", row.get("batch_id", 0)))
            composite_data[batch] = row

# Lead time CSV for alarm indices
lead_times = {}
lt_path = os.path.join(EARLY, "lead_time_per_batch.csv")
if os.path.exists(lt_path):
    with open(lt_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch = int(row["batch"])
            lead_times[batch] = {k: int(v) if v.lstrip("-").isdigit() else v
                                  for k, v in row.items()}

# ─── Plot Batch 93 in detail ─────────────────────────────────────────────────
batch_id = 93
mask = batch_ids == batch_id
s93 = scores[mask]
y93 = y[mask]
t93 = end_times[mask] if end_times is not None else np.arange(len(s93))
win_idx = np.arange(len(s93))

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top: raw anomaly score
ax = axes[0]
color_normal = "#2196F3"
color_fault = "#F44336"
colors = [color_fault if yi else color_normal for yi in y93]
ax.scatter(win_idx, s93, c=colors, s=12, alpha=0.8, zorder=3)
ax.axhline(threshold, color="darkorange", linewidth=2, linestyle="--", label=f"Static threshold ({threshold:.1f})")

# Mark key events
lt = lead_times.get(batch_id, {})
fault_w = lt.get("fault_window_idx", None)
trend_w = lt.get("trend_alarm_idx", None)
static_w = lt.get("static_alarm_idx", None)

if fault_w is not None:
    ax.axvline(fault_w, color="red", linewidth=2, linestyle=":", alpha=0.8, label=f"Fault onset (w={fault_w})")
if trend_w is not None and trend_w != fault_w:
    ax.axvline(trend_w, color="green", linewidth=2, linestyle="-.", alpha=0.8, label=f"Trend alarm (w={trend_w})")
if static_w is not None and static_w != fault_w:
    ax.axvline(static_w, color="darkorange", linewidth=2, linestyle="-", alpha=0.6, label=f"Static alarm (w={static_w})")

normal_patch = mpatches.Patch(color=color_normal, label="Normal windows")
fault_patch = mpatches.Patch(color=color_fault, label="Fault windows (ground truth)")
ax.legend(handles=[normal_patch, fault_patch] + ax.get_lines(), loc="upper left", fontsize=9)
ax.set_ylabel("Anomaly Score", fontsize=11)
ax.set_title(f"Batch 93 — GRU-AE v2 Anomaly Score vs Window Index\n"
             f"Fault onset: window {fault_w} | Trend alarm: window {trend_w} | "
             f"Static alarm: window {static_w}", fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate lead times
if fault_w is not None and trend_w is not None:
    lead = fault_w - trend_w
    ax.annotate(f"True lead: {lead} windows\n({lead * 5} timesteps)",
                xy=(trend_w, threshold * 0.7),
                xytext=(trend_w + 5, threshold * 0.9),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=9, color="green")

# Bottom: score derivative (trend signal)
ax2 = axes[1]
deriv = np.gradient(s93)
ax2.plot(win_idx, deriv, color="#9C27B0", linewidth=1.2, label="Score derivative (trend signal)")
ax2.axhline(0, color="black", linewidth=0.8)
if trend_w is not None:
    ax2.axvline(trend_w, color="green", linewidth=2, linestyle="-.", alpha=0.8, label=f"Trend alarm (w={trend_w})")
if fault_w is not None:
    ax2.axvline(fault_w, color="red", linewidth=2, linestyle=":", alpha=0.8, label=f"Fault onset (w={fault_w})")
ax2.set_xlabel("Window Index (stride=5 timesteps each)", fontsize=11)
ax2.set_ylabel("dScore/dWindow", fontsize=11)
ax2.set_title("Score Derivative — Basis for Trend Alarm", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "batch93_score_curve.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ─── Plot all 10 faulty batches ───────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
faulty_batches = list(range(91, 101))

for i, bid in enumerate(faulty_batches):
    ax = axes[i // 2][i % 2]
    mask = batch_ids == bid
    sb = scores[mask]
    yb = y[mask]
    wb = np.arange(len(sb))

    c = [color_fault if yi else color_normal for yi in yb]
    ax.scatter(wb, sb, c=c, s=8, alpha=0.7)
    ax.axhline(threshold, color="darkorange", linewidth=1.5, linestyle="--", alpha=0.9)

    lt = lead_times.get(bid, {})
    fw = lt.get("fault_window_idx", None)
    tw = lt.get("trend_alarm_idx", None)
    sw = lt.get("static_alarm_idx", None)

    if fw is not None:
        ax.axvline(fw, color="red", linewidth=1.5, linestyle=":", alpha=0.8)
    if tw is not None and tw != fw:
        ax.axvline(tw, color="green", linewidth=1.5, linestyle="-.", alpha=0.8)

    lead_info = ""
    if fw is not None and tw is not None:
        lead = fw - tw
        lead_info = f"\nTrue lead: {lead}w" if lead > 0 else f"\nNo early warn"
    ax.set_title(f"Batch {bid}{lead_info}", fontsize=9)
    ax.set_xlabel("Window", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.grid(True, alpha=0.3)

# Legend
normal_patch = mpatches.Patch(color=color_normal, label="Normal")
fault_patch = mpatches.Patch(color=color_fault, label="Fault")
import matplotlib.lines as mlines
trend_line = mlines.Line2D([], [], color="green", linestyle="-.", linewidth=1.5, label="Trend alarm")
onset_line = mlines.Line2D([], [], color="red", linestyle=":", linewidth=1.5, label="Fault onset")
thr_line = mlines.Line2D([], [], color="darkorange", linestyle="--", linewidth=1.5, label="Static threshold")
fig.legend(handles=[normal_patch, fault_patch, onset_line, trend_line, thr_line],
           loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0))

plt.suptitle("GRU-AE v2 — All Faulty Batches (91–100) Score Curves", fontsize=13, y=1.01)
plt.tight_layout()
out_path2 = os.path.join(FIG_DIR, "all_faulty_batches_scores.png")
plt.savefig(out_path2, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path2}")

# ─── Print diagnostic summary ────────────────────────────────────────────────
print("\n=== BATCH 93 DIAGNOSTIC ===")
print(f"Total windows: {len(s93)}")
print(f"Fault onset window: {fault_w}")
print(f"Trend alarm window: {trend_w}")
print(f"Static alarm window: {static_w}")
print(f"\nScore at window {trend_w} (trend alarm): {s93[trend_w]:.2f}")
print(f"Score at window {fault_w} (fault onset): {s93[fault_w]:.2f}")
print(f"Mean score in windows 0-{trend_w}: {s93[:trend_w+1].mean():.2f}")
print(f"Mean score in windows {trend_w}-{fault_w}: {s93[trend_w:fault_w+1].mean():.2f}")
print(f"Static threshold: {threshold:.2f}")
print(f"\nIs score at trend alarm window above threshold? {s93[trend_w] >= threshold}")
print(f"Score values around trend alarm (w=8 to w=15):")
for w in range(max(0, trend_w-3), min(len(s93), trend_w+8)):
    marker = " <<<" if w == trend_w else ""
    print(f"  w={w:3d}: score={s93[w]:.4f}{marker}")
