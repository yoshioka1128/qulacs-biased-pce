import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pltutils

files = [
#    "outputs/power_opt/pce_greedy_summary_rate0.1_bias_x.csv",
    "outputs/power_opt/pce_greedy_summary_rate0.1_nobias.csv",
    "outputs/power_opt/pce_greedy_summary_rate0.1_bias_y.csv",
]

# 見た目の設定
plt.figure()

# マーカーをファイルごとに変える（任意）
markers = ["o", "^", "s"]
labels = ["PCE", "PCE w/ bias", ""]

for f, mk, label in zip(files, markers, labels):
    df = pd.read_csv(f)
    label_base = Path(f).stem  # ファイル名（拡張子なし）

    # cost_wo_pp
    plt.scatter(df["nodes"], df["cost_wo_pp"],
                alpha=0.8, s=45, marker=mk, label=label)

    # cost
    plt.scatter(df["nodes"], df["cost"],
                label=f"{label_base}: cost",
                alpha=0.8, s=45, marker=mk, edgecolors="k", linewidths=0.3)

plt.xlabel("nodes")
plt.ylabel("normalized cost")
plt.title("rate 0.1")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(fontsize=8, ncol=2)

plt.xscale('log')
#plt.yscale('log')
plt.ylim(0.0, 4.0*1.0e-5)
plt.xlim(10, 1e4)
# 値が非常に小さいので、対数スケールにしたい場合は以下を有効化
# plt.yscale("symlog", linthresh=1e-12)  # 0や負値があっても比較的安全

plt.tight_layout()
plt.show()
