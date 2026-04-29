from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pltutils

BASE = Path("outputs/power_opt")

files = [
    BASE / "csv/pce_greedy_averaged_summary_rate0.1_nobias.csv",
    BASE / "csv/pce_greedy_averaged_summary_rate0.1_bias_y.csv",
    BASE / "csv/greedy_allzero_averaged_summary_rate0.1.csv",  # ★ 追加
]

markers = ["o", "^", "x"]  # ★ 追加
labels = ["PCE", "BA-PCE", "greedy (all-zero)"]

# 色を明示的に管理
colors = ["tab:blue", "tab:orange", "green"]

plt.figure()

for f, mk, label, color in zip(files, markers, labels, colors):
    df = pd.read_csv(f)

    if "cost_wo_pp" in df.columns:
        # PCE系（before/afterあり）
        plt.scatter(
            df["nodes"], df["cost_wo_pp"],
            s=100,
            facecolors="none",
            marker=mk,
            linewidths=1.5,
            edgecolor=color,
            label=f"{label} w/o pp"
        )

        plt.scatter(
            df["nodes"], df["cost"],
            s=100,
            marker=mk,
            color=color,
            label=f"{label}"
        )
    else:
        # ★ all-zero greedy（afterのみ）
        plt.scatter(
            df["nodes"], df["cost"],
            s=100,
            marker=mk,
            color=color,
            label=label
        )

plt.xlabel(r"problem size $m$")
plt.ylabel(r"normalized cost gap $\Delta C_T/W_T$")

plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.legend()

plt.xscale("log")
plt.yscale("log")
plt.xlim(10, 1e4)
plt.ylim(5e-6, 1.0e-3)

plt.tight_layout()

# 保存
out_path = BASE / "figures/png/normalized_cost_greedy_vs_pce_rate0.1.png"
out_path = BASE / "figures/pdf/normalized_cost_greedy_vs_pce_rate0.1.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)

print(f"saved -> {out_path}")

plt.show()
