import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pltutils

# =========================
# CSV読み込み
# =========================
base_path = "outputs/power_opt/csv/"

df_allzero = pd.read_csv(base_path + "greedy_allzero_averaged_summary_all.csv")
df_random  = pd.read_csv(base_path + "greedy_random_averaged_summary_all.csv")
df_pce     = pd.read_csv(base_path + "pce_greedy_averaged_summary_all.csv")

# =========================
# フィルタ (nodes == 756)
# =========================
df_allzero = df_allzero[df_allzero["nodes"] == 756]
df_random  = df_random[df_random["nodes"] == 756]
df_pce     = df_pce[(df_pce["nodes"] == 756) & (df_pce["bias_mode"] == "nobias")]

# =========================
# plot
# =========================
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom = 0.11)

# --- greedy_allzero ---
greedyzero =ax.scatter(
    df_allzero["rate"],
    df_allzero["cost"],
    label="greedy (all-zero)",
    color="green",
    linewidth=2.0,
    s=100,
    marker="x"
)

# --- greedy_random (error bar付き) ---
greedyran= ax.errorbar(
    df_random["rate"],
    df_random["cost"],
    markersize=10,
    linewidth=2.0,
    yerr=df_random["std"],
    fmt="+",
    capsize=3,
    color="black",
    label="greedy (random)"
)

# --- pce_greedy ---
## cost_wo_pp（solid）
#ax.scatter(
#    df_pce["rate"],
#    df_pce["cost_wo_pp"],
#    label="pce (wo pp)",
#    marker="s"
#)

# cost（open marker）
pce = ax.scatter(
    df_pce["rate"],
    df_pce["cost"],
    label="PCE simulation",
#    facecolors="none",
    edgecolors="C0",
    marker="o",
    linewidth=2.0,
    s=100
)

# =========================
# 装飾
# =========================
#ax.set_xticks(np.arange(0.1, 0.6, 0.1))
ax.set_xlabel(r"target level parameter $\eta$")
ax.set_ylabel(r"normalized cost gap $\Delta C_T/W_T$")
ax.set_yscale('log')
ax.set_xlim(right = 0.42)
ax.set_ylim(1.0e-5, 3.0e-3)

#ax.legend()
ax.legend([greedyzero, greedyran, pce], ["greedy (all-zero)", "greedy (random)", "PCE simulation"])

plt.grid(True, which="both", linestyle=":", linewidth=0.5)

out_path = "outputs/power_opt/figures/pdf/cost_vs_rate.pdf"
fig.savefig(out_path)

#plt.tight_layout()
plt.show()
