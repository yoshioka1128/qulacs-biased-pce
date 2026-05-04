from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pltutils

BASE = Path("outputs/power_opt")

# =========================
# CSV読み込み
# =========================
df_pce_all = pd.read_csv(BASE / "csv/pce_greedy_averaged_summary_all.csv")
df_allzero_all = pd.read_csv(BASE / "csv/greedy_allzero_averaged_summary_all.csv")

# rate指定
rate = 0.1
df_pce_all = df_pce_all[df_pce_all["rate"] == rate]
df_allzero = df_allzero_all[df_allzero_all["rate"] == rate]

# =========================
# データ分割
# =========================
df_pce    = df_pce_all[df_pce_all["bias_mode"] == "nobias"]
df_ba_pce = df_pce_all[df_pce_all["bias_mode"] == "bias_y"]

datasets = [
    (df_pce,    "o", "PCE", "tab:blue"),
    (df_ba_pce, "^", "BA-PCE", "tab:orange"),
    (df_allzero, "x", "greedy (all-zero)", "green"),
]

# =========================
# plot
# =========================
fig, ax = plt.subplots()

for df_i, mk, label, color in datasets:

    if "cost_wo_pp" in df_i.columns:
        # --- before (open marker) ---
        ax.scatter(
            df_i["nodes"], df_i["cost_wo_pp"],
            s=100,
            facecolors="none",
            marker=mk,
            linewidths=1.5,
            edgecolors=color,
            label=f"{label} w/o pp"
        )

        # --- after (solid marker) ---
        ax.scatter(
            df_i["nodes"], df_i["cost"],
            s=100,
            marker=mk,
            color=color,
            label=f"{label}"
        )
    else:
        # --- greedy all-zero ---
        ax.scatter(
            df_i["nodes"], df_i["cost"],
            s=100,
            marker=mk,
            color=color,
            label=label
        )

# =========================
# 装飾
# =========================
ax.set_xlabel(r"problem size $m$")
ax.set_ylabel(r"normalized cost gap $\Delta C_T/W_T$")

ax.grid(True, which="both", linestyle=":", linewidth=0.5)
ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 1e4)
ax.set_ylim(5e-6, 1.0e-3)

fig.tight_layout()

# =========================
# 保存
# =========================
out_path = BASE / "figures/pdf/normalized_cost_greedy_vs_pce_rate0.1.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(out_path)

print(f"saved -> {out_path}")

plt.show()
