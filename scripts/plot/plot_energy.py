import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from src.analysis.loader import load_data
from scripts.plot.plot_core import plot_energy, DEFAULT_BETAS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=18)
    parser.add_argument("--mode", choices=["mean", "min", "band"], default="band")
    args = parser.parse_args()

    file_map = {
        18: "time1_nT24_rate0.2_18nodes_4qubits_2body_ninit5_depth5_all2all_methodBFGS_iseed42",
        60: "time1_nT24_rate0.2_60nodes_6qubits_3body_ninit5_depth5_all2all_methodBFGS_iseed42",
        210: "time1_nT24_rate0.2_210nodes_8qubits_4body_ninit5_depth5_all2all_methodBFGS_iseed42",
        756: "time1_nT24_rate0.2_756nodes_10qubits_5body_ninit5_depth5_all2all_methodBFGS_iseed42",
    }

    BASE_DIR = Path(__file__).resolve().parents[2]
    file = file_map[args.m]

    DATA_DIR_NO_BIAS = BASE_DIR / "outputs" / "power_opt" / file
    DATA_DIR_WITH_BIAS = BASE_DIR / "outputs" / "power_opt" / f"{file}_bias"

    SAVE_DIR = BASE_DIR / "outputs" / "power_opt" / "figures"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    energy_nb, loss_nb = load_data(DATA_DIR_NO_BIAS, use_bias=False)
    energy_wb, loss_wb = load_data(DATA_DIR_WITH_BIAS, use_bias=True)
    aggregation = args.mode
    save_path = SAVE_DIR / f"{aggregation}_energy.png"

    plot_energy(
        energy_nb,
        energy_wb,
        DEFAULT_BETAS,
        aggregation=aggregation,
        save_path=save_path
       )

if __name__ == "__main__":
    main()
