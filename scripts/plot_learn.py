import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "outputs" / "power_opt" / \
    "time1_nT24_rate0.2_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42_new"
#    "time1_nT24_rate0.2_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42_new"

# 保存先（指定通り）
SAVE_DIR = BASE_DIR / "outputs" / "power_opt" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"alphasc([-\d\.]+)_beta([-\d\.]+)")


def load_data(use_bias: bool):
    """biasあり/なしを切り替えてデータ読み込み"""
    energy_data = defaultdict(lambda: defaultdict(list))
    loss_data = defaultdict(lambda: defaultdict(list))

    for file in DATA_DIR.glob("results_backprop_*.json"):
        fname = file.name

        # --- biasフィルタ ---
        if use_bias:
            if "bias" not in fname:
                continue
        else:
            if "bias" in fname:
                continue

        # --- パラメータ抽出 ---
        match = pattern.search(fname)
        if not match:
            continue

        alphasc = float(match.group(1))
        beta = float(match.group(2))

        # --- JSON読み込み ---
        with open(file, "r") as f:
            js = json.load(f)

        energy = js["Calculated Minimum Energy [norm, row]"][0]
        loss = js["Corresponding loss function"]

        energy_data[beta][alphasc].append(energy)
        loss_data[beta][alphasc].append(loss)

    return energy_data, loss_data


def plot_data(energy_data, loss_data, title, save_path):
    plt.figure()

    target_betas = [-0.1, 0.0, 0.1, 0.2]

    for beta in target_betas:
        if beta not in energy_data:
            continue

        alphasc_list = sorted(energy_data[beta].keys())

        xs = []
        energy_mean = []
        loss_mean = []

        for a in alphasc_list:
            xs.append(a)
            energy_mean.append(np.mean(energy_data[beta][a]))
            loss_mean.append(np.mean(loss_data[beta][a]))

        # 実線：energy
        plt.plot(xs, energy_mean, marker="o", linestyle="-",
                 label=f"Energy (beta={beta})")

        # 破線：loss
        plt.plot(xs, loss_mean, marker="x", linestyle="--",
                 label=f"Loss (beta={beta})")

    plt.xlabel("alphasc")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(save_path)

# =========================
# 実行
# =========================

# --- biasなし ---
energy_data, loss_data = load_data(use_bias=False)
plot_data(
    energy_data,
    loss_data,
    title="Backprop (no bias)",
    save_path=SAVE_DIR / "backprop_no_bias.png"
)

# --- biasあり ---
energy_data, loss_data = load_data(use_bias=True)
plot_data(
    energy_data,
    loss_data,
    title="Backprop (with bias)",
    save_path=SAVE_DIR / "backprop_with_bias.png"
)

# =========================
# 3つ目：Energy比較（biasあり vs なし）
# =========================

energy_no_bias, _ = load_data(use_bias=False)
energy_with_bias, _ = load_data(use_bias=True)

plt.figure()

target_betas = [-0.1, 0.0, 0.1, 0.2]

for beta in target_betas:

    # --- biasなし ---
    if beta in energy_no_bias:
        alphasc_list = sorted(energy_no_bias[beta].keys())

        xs = []
        ys = []

        for a in alphasc_list:
            xs.append(a)
            ys.append(np.mean(energy_no_bias[beta][a]))

        plt.plot(xs, ys, linestyle="-", marker="o",
                 label=f"No bias (beta={beta})")

    # --- biasあり ---
    if beta in energy_with_bias:
        alphasc_list = sorted(energy_with_bias[beta].keys())

        xs = []
        ys = []

        for a in alphasc_list:
            xs.append(a)
            ys.append(np.mean(energy_with_bias[beta][a]))

        plt.plot(xs, ys, linestyle="--", marker="x",
                 label=f"With bias (beta={beta})")


plt.xlabel("alphasc")
plt.ylabel("Energy")
plt.ylim(0.0, 0.003)
plt.title("Energy comparison (bias vs no bias)")
plt.legend()
plt.grid()

plt.savefig(SAVE_DIR / "energy_comparison_bias_vs_no_bias.png")
plt.show()


plt.show()
