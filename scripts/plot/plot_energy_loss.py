import argparse
from pathlib import Path

from src.analysis.loader import load_data
from scripts.plot.plot_core import plot_energy_loss_by_beta, DEFAULT_BETAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=18)
    parser.add_argument("--model", default="power_opt")
    args = parser.parse_args()

    file_map = {
        18: "...",
        60: "...",
        210: "...",
        756: "...",
    }

    BASE_DIR = Path(__file__).resolve().parents[2]
    file = file_map[args.m]

    BASE_OUTPUT = BASE_DIR / "outputs" / args.model

    DATA_DIR = BASE_OUTPUT / file

    SAVE_DIR = BASE_OUTPUT / "figures"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    energy_data, loss_data = load_data(DATA_DIR, use_bias=False)

    plot_energy_loss_by_beta(
        energy_data,
        loss_data,
        DEFAULT_BETAS,
        SAVE_DIR / f"{args.model}_energy_loss.png"
    )


if __name__ == "__main__":
    main()
