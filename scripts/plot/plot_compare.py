# plot_compare.py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.analysis.loader import load_data
from src.analysis.aggregator import aggregate

target_betas = [-0.1, 0.0, 0.1, 0.2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=18)
    args = parser.parse_args()

    file_map = {
        18: "time1_nT24_rate0.2_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42",
        60: "time1_nT24_rate0.2_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42",
        210: "time1_nT24_rate0.2_210nodes_8qubits_4body_ninit5_depth5_all2all_methodBFGS_iseed42",
        756: "time1_nT24_rate0.2_756nodes_10qubits_5body_ninit5_depth5_all2all_methodBFGS_iseed42",
    }

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "outputs" / "power_opt" / file_map[args.nodes]
    SAVE_DIR = BASE_DIR / "outputs" / "power_opt" / "figures"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    energy_data, _ = load_data(DATA_DIR, use_bias=False)

    plt.figure()

    for beta in target_betas:
        if beta not in energy_data:
            continue

        xs, mean_vals, min_vals = [], [], []

        for a in sorted(energy_data[beta].keys()):
            if "no_reg" not in energy_data[beta][a]:
                continue

            vals = energy_data[beta][a]["no_reg"]

            xs.append(a)
            mean_vals.append(aggregate(vals, "mean"))
            min_vals.append(aggregate(vals, "min"))

        plt.plot(xs, mean_vals, linestyle="-", label=f"mean beta={beta}")
        plt.plot(xs, min_vals, linestyle="--", label=f"min beta={beta}")

    plt.title("Mean vs Min comparison")
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_DIR / "compare_mean_min.png")
    plt.show()

if __name__ == "__main__":
    main()
