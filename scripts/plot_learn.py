import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

marker_map = {
    -0.1: "o",   # circle
    0.0: "s",    # square
    0.1: "^",    # triangle
    0.2: "D",    # diamond
}

target_betas = [-0.1, 0.0, 0.1, 0.2]
# target_betas = [-0.1, 0.0, 0.5, 1.0, 1.5, 2.0]

nodes = int(input('input nodes: ') or 18)
if nodes ==18: file = "time1_nT24_rate0.2_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42"
elif nodes == 60: file ="time1_nT24_rate0.2_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42"
elif nodes == 210: file = "time1_nT24_rate0.2_210nodes_8qubits_4body_ninit5_depth5_all2all_methodBFGS_iseed42"
elif nodes == 756: file = "time1_nT24_rate0.2_756nodes_10qubits_5body_ninit5_depth5_all2all_methodBFGS_iseed42"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR_NO_BIAS = BASE_DIR / "outputs" / "power_opt" / file
DATA_DIR_WITH_BIAS = BASE_DIR / "outputs" / "power_opt" / f"{file}_bias"

# 保存先（指定通り）
SAVE_DIR = BASE_DIR / "outputs" / "power_opt" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"alphasc([-\d\.]+)_beta([-\d\.]+)")

def load_data(data_dir: Path, use_bias: bool):
    # beta → alphasc → reg_type → list
    energy_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    loss_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file in data_dir.glob("results_backprop*.json"):
        fname = file.name

        # --- biasフィルタ ---
        if use_bias:
            if "results_backprop_bias_" not in fname:
                continue
        else:
            if "results_backprop_bias_" in fname:
                continue

        # --- reg_type判定 ---
        if use_bias:
            if "_reg_typex" in fname:
                reg_type = "x"
            elif "_reg_typey" in fname:
                reg_type = "y"
            else:
                reg_type = "unknown"
        else:
            reg_type = "no_reg"

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

        energy_data[beta][alphasc][reg_type].append(energy)
        loss_data[beta][alphasc][reg_type].append(loss)

    return energy_data, loss_data


def plot_data_by_regtype(energy_data, loss_data, title, save_path):
    plt.figure()

    target_betas = [-0.1, 0.0, 0.1, 0.2]

    # 色とスタイル
    color_map = {
        "no_reg": "black",
        "x": "blue",
        "y": "red",
        "unknown": "gray"
    }

    linestyle_map = {
        "energy": "-",
        "loss": "--"
    }

    for beta in target_betas:
        if beta not in energy_data:
            continue

        alphasc_list = sorted(energy_data[beta].keys())

        for reg_type in ["no_reg", "x", "y"]:
            xs = []
            energy_mean = []
            loss_mean = []

            for a in alphasc_list:
                if reg_type not in energy_data[beta][a]:
                    continue

                xs.append(a)
                energy_mean.append(np.mean(energy_data[beta][a][reg_type]))
                loss_mean.append(np.mean(loss_data[beta][a][reg_type]))

            if not xs:
                continue

            marker = marker_map[beta]

            # Energy（実線）
            plt.plot(
                xs, energy_mean,
                marker=marker,
                linestyle="-",
                color=color_map[reg_type],
                label=f"Energy (beta={beta}, reg={reg_type})"
            )

            # Loss（破線）
            plt.plot(
                xs, loss_mean,
                marker=marker,
                linestyle="--",
                color=color_map[reg_type],
                label=f"Loss (beta={beta}, reg={reg_type})"
            )

    plt.xlabel("alphasc")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    

# --- biasなし ---
energy_data, loss_data = load_data(DATA_DIR_NO_BIAS, use_bias=False)
plot_data_by_regtype(
    energy_data,
    loss_data,
    title="Backprop (no bias)",
    save_path=SAVE_DIR / "backprop_no_bias.png"
)

# --- biasあり ---
energy_data, loss_data = load_data(DATA_DIR_WITH_BIAS, use_bias=True)
plot_data_by_regtype(
    energy_data,
    loss_data,
    title="Backprop (with bias, reg_type split)",
    save_path=SAVE_DIR / "backprop_with_bias_regtype.png"
)

energy_no_bias, _ = load_data(DATA_DIR_NO_BIAS, use_bias=False)
energy_with_bias, _ = load_data(DATA_DIR_WITH_BIAS, use_bias=True)

plt.figure()
for beta in target_betas:
    marker = marker_map[beta]

    # --- no bias ---
    if beta in energy_no_bias:
        alphasc_list = sorted(energy_no_bias[beta].keys())

        xs = []
        ys = []

        for a in alphasc_list:
            if "no_reg" not in energy_no_bias[beta][a]:
                continue
            xs.append(a)
            ys.append(np.mean(energy_no_bias[beta][a]["no_reg"]))

        plt.plot(xs, ys,
                 marker=marker,
                 linestyle="-",
                 color="black",
                 label=f"No bias (beta={beta})")
#        plt.plot(xs, ys, marker="o", linestyle="-",
#                 color="black", label=f"No bias (beta={beta})")

    # --- with bias (x, y) ---
    if beta in energy_with_bias:
        marker = marker_map[beta]
        alphasc_list = sorted(energy_with_bias[beta].keys())

        for reg_type, color in [("x", "blue"), ("y", "red")]:
            xs = []
            ys = []

            for a in alphasc_list:
                if reg_type not in energy_with_bias[beta][a]:
                    continue
                xs.append(a)
                ys.append(np.mean(energy_with_bias[beta][a][reg_type]))

            if xs:
                plt.plot(xs, ys,
                         marker=marker,
                         linestyle="--",
                         color=color,
                         label=f"With bias {reg_type} (beta={beta})")
#                plt.plot(xs, ys, marker="x", linestyle="--",
#                         color=color,
#                         label=f"With bias {reg_type} (beta={beta})")

plt.xlabel("alphasc")
plt.ylabel("Energy")
plt.ylim(0.0, 0.01)
plt.title("Energy comparison (bias vs no bias, reg_type split)")
plt.legend()
plt.grid()

plt.savefig(SAVE_DIR / "energy_comparison_regtype.png")

# --- betaごとの色指定 ---
beta_color_map = {
    -0.1: "C2",
    0.0: "C0",
    0.1: "C1",
    0.2: "C3",
}

def plot_by_dataset(energy_data, loss_data, reg_type, title, save_path):
    plt.figure()

    for beta in target_betas:
        if beta not in energy_data:
            continue

        alphasc_list = sorted(energy_data[beta].keys())

        xs = []
        energy_mean = []
        loss_mean = []

        for a in alphasc_list:
            if reg_type not in energy_data[beta][a]:
                continue

            xs.append(a)
            energy_mean.append(np.mean(energy_data[beta][a][reg_type]))
            loss_mean.append(np.mean(loss_data[beta][a][reg_type]))

        if not xs:
            continue

        color = beta_color_map[beta]

        # energy（実線）
        plt.plot(xs, energy_mean,
                 linestyle="-",
                 marker="o",
                 color=color,
                 label=f"Energy beta={beta}")

        # loss（破線）
        plt.plot(xs, loss_mean,
                 linestyle="--",
                 marker="x",
                 color=color,
                 label=f"Loss beta={beta}")

    plt.xlabel("alphasc")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(save_path)


# --- No bias ---
energy_nb, loss_nb = load_data(DATA_DIR_NO_BIAS, use_bias=False)
plot_by_dataset(
    energy_nb, loss_nb,
    reg_type="no_reg",
    title="No bias (beta color, energy/loss style)",
    save_path=SAVE_DIR / "no_bias_beta_color.png"
)

# --- with bias x ---
energy_wb, loss_wb = load_data(DATA_DIR_WITH_BIAS, use_bias=True)
plot_by_dataset(
    energy_wb, loss_wb,
    reg_type="x",
    title="With bias x (beta color, energy/loss style)",
    save_path=SAVE_DIR / "with_bias_x_beta_color.png"
)

# --- with bias y ---
plot_by_dataset(
    energy_wb, loss_wb,
    reg_type="y",
    title="With bias y (beta color, energy/loss style)",
    save_path=SAVE_DIR / "with_bias_y_beta_color.png"
)

plt.show()
