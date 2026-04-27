import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.analysis.loader import load_data
from scripts.plot.plot_core import plot_cost, DEFAULT_BETAS


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--m", type=int, default=18, help="problem size")
    parser.add_argument("--mode", choices=["mean", "min", "band"], default="band",
                        help="aggregation mode")
    parser.add_argument("--model", choices=["no_bias", "bias_x", "bias_y", "all"], default="all",
                        help="plot target model")
    args = parser.parse_args()

    file_map = {
        18: "time1_nT24_rate0.1_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42",
        60: "time1_nT24_rate0.1_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42",
        210: "time1_nT24_rate0.1_210nodes_8qubits_4body_ninit5_depth5_all2all_methodBFGS_iseed42",
        756: "time1_nT24_rate0.1_756nodes_10qubits_5body_ninit5_depth5_all2all_methodBFGS_iseed42",
        2772: "time1_nT24_rate0.1_2772nodes_12qubits_6body_ninit5_depth5_all2all_methodBFGS_iseed42",
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

    # =============================
    # ① 手法比較
    # =============================
    plot_cost(
        cost_nb,
        cost_wb,
        DEFAULT_BETAS,
        aggregation=aggregation,
        save_path=SAVE_DIR / f"{aggregation}_cost_method.png",
        loss_nb=loss_nb,
        loss_wb=loss_wb,
        mode="method",
    )

    # =============================
    # ② β色分け（モデル指定）
    # =============================
    if args.model == "all":
        models = ["bias_x", "bias_y", "no_bias"]
    else:
        models = [args.model]

    for model in models:
        plot_cost(
            cost_nb,
            cost_wb,
            DEFAULT_BETAS,
            aggregation=aggregation,
            save_path=SAVE_DIR / f"{aggregation}_cost_{model}.png",
            loss_nb=loss_nb,
            loss_wb=loss_wb,
            mode="beta",
            model=model,
        )

    plt.show()


if __name__ == "__main__":
    main()
    
