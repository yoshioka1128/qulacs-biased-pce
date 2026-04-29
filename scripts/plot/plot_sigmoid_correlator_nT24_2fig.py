import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.lines import Line2D
import pltutils
from config import NODE_CONFIG

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_exp_values(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r") as f:
        data = json.load(f)

    if "Corresponding exp value" not in data:
        raise KeyError('"Corresponding exp value" not found in JSON')

    return np.array(data["Corresponding exp value"])


# =========================
# Parameters
# =========================
rate = 0.5

betas = [-0.1, 0.0, 0.1]
strbp = ""

bound = 10.0
shots = 4001
delta_sigmoid = 0.01
bias = 0.0

bins_sigmoid = np.arange(0, 1 + delta_sigmoid, delta_sigmoid)
bin_centers_sigmoid = (bins_sigmoid[:-1] + bins_sigmoid[1:]) / 2

# =========================
# Figure（2段）
# =========================
fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), constrained_layout=True)

nodes_list = [18, 10296]

for ax_main, nodes in zip(axes, nodes_list):
    # =========================
    # パラメータ再設定（ここ重要）
    # =========================
    cfg = NODE_CONFIG[nodes]

    iinit_list = cfg['chbetaiinit']
    alphasc = cfg['alphasc']
    strbp = cfg['strbp']
    calpha = cfg['calpha']
    n_qubits = cfg['n_qubits']
    k = cfg['k']

    if nodes == 18: calpha=50
    elif nodes == 210: calpha=50
    elif nodes == 10296: calpha = 1.0

    if nodes == 18:
        calpha = 50
    elif nodes == 210:
        calpha = 50
    elif nodes == 10296:
        calpha = 1.0

    alpha = alphasc * n_qubits ** np.floor(k / 2)

    shots2 = shots // calpha
    if shots2 % 2 == 0:
        shots2 += 1

    klist = np.arange(shots2 + 1)
        
    klist = np.arange(shots2 + 1)
    x = (2 * klist - shots2) / shots2
    delta = x[1] - x[0]
    bins = np.concatenate([[x[0] - delta / 2], x[:-1] + delta / 2, [x[-1] + delta / 2]])
    bin_centers = np.array(x)

    bins_sigmoid = np.arange(0, 1 + delta_sigmoid, delta_sigmoid)
    bin_centers_sigmoid = (bins_sigmoid[:-1] + bins_sigmoid[1:]) / 2

    # =========================
    # (main) sigmoid
    # =========================
    for iinit, beta in zip(iinit_list, betas):
        if beta == -0.1:
            continue

        file_path = (
            f"outputs/power_opt/time1_nT24_rate{rate}_{nodes}nodes_{n_qubits}qubits_{k}body_ninit5_"
            "depth5_all2all_methodBFGS_iseed42_new/"
            f"results{strbp}_alpha{alpha}_beta{beta}_init{iinit}_iseed42.json"
        )

        exp_values = - load_exp_values(file_path) # bugfix
        scaled_values = alpha * exp_values
        sigmoid_values = sigmoid(2.0 * scaled_values + bias)

        counts_sigmoid, _ = np.histogram(sigmoid_values, bins=bins_sigmoid)

        ax_main.bar(
            bin_centers_sigmoid,
            counts_sigmoid,
            width=delta_sigmoid,
            alpha=0.4,
            label=rf"$\beta={beta}$" # bugfix
        )

    ax_main.set_xlim([0, 1])
    ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_main.yaxis.set_label_coords(-0.1, 0.5)

    if nodes == 18: ax_main.set_ylim([0, 20])
    ax_main.set_ylabel(r"$\textrm{frequency}$")

    # ラベル
    label = r"$\textrm{(a)}\ m=18$" if nodes == 18 else r"$\textrm{(b)}\ m=10,296$"
    ax_main.text(0.02, 0.9, label, transform=ax_main.transAxes, fontweight="bold")

    # 凡例は上だけにするのが綺麗
#    if nodes == 18:
#        ax_main.legend()

    # =========================
    # inset
    # =========================
    ax_inset = inset_axes(
        ax_main,
        width="50%",
        height="46%",
        loc="center",
        bbox_to_anchor=(-0.0, 0.09, 1, 1),
        bbox_transform=ax_main.transAxes,
        borderpad=0
    )

    for iinit, beta in zip(iinit_list, betas):
        if beta == -0.1:
            continue

        file_path = (
            f"outputs/power_opt/time1_nT24_rate{rate}_{nodes}nodes_{n_qubits}qubits_{k}body_ninit5_"
            "depth5_all2all_methodBFGS_iseed42_new/"
            f"results{strbp}_alpha{alpha}_beta{beta}_init{iinit}_iseed42.json"
        )

        exp_values = - load_exp_values(file_path) # bugfix
        counts, _ = np.histogram(exp_values, bins=bins)

        ax_inset.bar(
            bin_centers,
            counts,
            width=delta,
            alpha=0.4,
            label=rf"$\beta={beta}$" # bugfix
        )

    # inset設定
    if nodes in [60, 210]:
        ax_inset.set_xlim([-1, 1])
    elif nodes == 18:
        ax_inset.set_xlim([-0.8, 0.8])
        ax_inset.set_ylim([0.0, 10])
    else:
        ax_inset.set_xlim([-0.03, 0.03])
        ax_inset.set_ylim([0.0, 600])
    ax_inset.set_xlabel(
#    r"Pauli correlator $\langle \psi(\boldsymbol{\theta})|\Pi_i^{\mathrm{(k)}}|\psi(\boldsymbol{\theta}) \rangle_{\boldsymbol{\theta}}$",
    r"$\textrm{Pauli\ correlator}\ \langle \Pi_i^{\mathrm{(k)}} \rangle_{\boldsymbol{\theta}^*}$",
        fontsize=14
    )

    ax_inset.tick_params(labelsize=12)
   
leg = axes[0].legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    fontsize=13
)

# 共通xlabel（下だけ）
for i in range(2):
    axes[i].set_xlabel(r"$\textrm{continuous variable}\ y_i(\boldsymbol{\theta}^*)$")
    axes[i].yaxis.set_label_coords(-0.1, 0.5)

# 保存
plt.savefig("outputs/power_opt/sigmoid_correlator_nT24_2fig.pdf")
plt.savefig("outputs/power_opt/sigmoid_correlator_nT24_2fig.png")
plt.show()

