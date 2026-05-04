import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.analysis.loader import load_data
from scripts.plot.plot_core import plot_cost, DEFAULT_BETAS
import pltutils

from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0,0))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--m", type=int, default=18, help="problem size")
    parser.add_argument("--bias_mode", choices=["nobias", "bias_x", "bias_y", "two", "all"], default="two",
                        help="plot target bias_mode")
    parser.add_argument("--rate", type=float, default=0.1, help="rate of target pwower")
    parser.add_argument("--mode", choices=["mean", "min", "band"], default="band",
                        help="aggregation mode")
    args = parser.parse_args()
    rate = args.rate
    file_map = {
        18: f"time1_nT24_rate{rate}_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42",
        60: f"time1_nT24_rate{rate}_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42",
        210: f"time1_nT24_rate{rate}_210nodes_8qubits_4body_ninit5_depth5_all2all_methodBFGS_iseed42",
        756: f"time1_nT24_rate{rate}_756nodes_10qubits_5body_ninit5_depth5_all2all_methodBFGS_iseed42",
        2772: f"time1_nT24_rate{rate}_2772nodes_12qubits_6body_ninit5_depth5_all2all_methodBFGS_iseed42",
    }

    BASE_DIR = Path(__file__).resolve().parents[2]
    file = file_map[args.m]

    DATA_DIR = BASE_DIR / "outputs" / "power_opt" / file
    SAVE_DIR = BASE_DIR / "outputs" / "power_opt" / "figures"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # --- load ---
    cost_nb, loss_nb = load_data(DATA_DIR, use_bias=False)
    cost_wb, loss_wb = load_data(DATA_DIR, use_bias=True)

    aggregation = args.mode

    rates = [0.1, 0.5]

    if args.bias_mode == "all":
        bias_modes = ["bias_x", "bias_y", "nobias"]
    elif args.bias_mode == "two":
        bias_modes = ["bias_y", "nobias"]
    else:
        bias_modes = [args.bias_mode]

    nrows = len(rates)
    ncols = len(bias_modes)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=True,
        sharey=False,
       )
    fig.subplots_adjust(
        left=0.1,
        right=0.98,
        top=0.95,
        bottom=0.08,
        wspace=0.15,
        hspace=0.2,
    )

    # axesを2次元配列として扱うための保険
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    labels = ["(a) OB-PCE", "(b) PCE", "(c) OB-PCE", "(d) PCE"]
    for i, rate in enumerate(rates):

        # file名をrateごとに作る
        file = file_map[args.m].replace(f"rate{args.rate}", f"rate{rate}")
        DATA_DIR = BASE_DIR / "outputs" / "power_opt" / file

        # --- load ---
        cost_nb, loss_nb = load_data(DATA_DIR, use_bias=False)
        cost_wb, loss_wb = load_data(DATA_DIR, use_bias=True)

        for j, bias_mode in enumerate(bias_modes):

            ax = axes[i][j]

            plot_cost(
                ax,
                cost_nb,
                cost_wb,
                DEFAULT_BETAS,
                aggregation=aggregation,
    #            loss_nb=loss_nb,
    #            loss_wb=loss_wb,
                mode="beta",
                bias_mode=bias_mode,
                show_label=(i == 0 and j == 0),
                add_legend=(i == 0 and j == 0),   # ← 左上だけ
                xmin=0.4,
                xmax=3.1
            )

            # タイトル
            idx = i * ncols + j
            if idx == 1:
                x = 0.15
            else:
                x = 0.02
            ax.text(
                x, 0.95,          
                labels[idx] ,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=1.0,
                    pad=1.2
                )
            )
            if i == 0:
                ax.tick_params(labelbottom=True)
            
#            if i == 0:
#                collabel = bias_mode
#                if bias_mode == 'bias_y': collabel = "OB-PCE"
#                elif bias_mode == 'nobias': collabel = "PCE"
#                ax.set_title(f"{collabel}")
#
            if j == 0:
                ax.set_ylabel(rf"$\eta$ = {rate}", labelpad=10)

#            ax.set_xlabel(r"$\alpha_{\rm sc}$")                
            if i == 0: ax.set_ylim(0.0, 3.0e-4)
            elif i==1: ax.set_ylim(0.0, 4.0e-3)
#            if i == 0: ax.set_ylim(1.0e-5, 1.0e-3)
#            elif i==1: ax.set_ylim(1.0e-4, 1.0e-2)
            ax.yaxis.set_major_formatter(formatter)

    # --- 凡例を1つにまとめる ---
    handles, labels = axes[0][0].get_legend_handles_labels()
    
    fig.supylabel(r"normalized cost gap $\Delta C_T/W_T$ w/o PP", x=0.0015)

    fig.savefig(SAVE_DIR / "pdf" / "cost_lean_3model.pdf")
    plt.show()

if __name__ == "__main__":
    main()
    
