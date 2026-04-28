# main.py
import argparse
from src.app.runner import run
from src.config.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--m", type=int, default=60, help="problem size")
    parser.add_argument("--batch", action="store_true", help="run in batch mode")
    parser.add_argument("--mode", type=str, choices=["nobias", "bias_x", "bias_y"], default="nobias",
                        help="regularization mode")
    parser.add_argument("--readmode", action="store_true", help="read existing result files")
    parser.add_argument("--itime", type=int, default=1, help="time for start")
    parser.add_argument("--nT", type=int, default=24, help="number of time steps")
    parser.add_argument("--rate", type=float, default=0.1, help="rate of target pwower")

    # 上書き用（任意）
    parser.add_argument("--n_qubits", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--alphasc", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--type_ansatz", type=str, default="all2all")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.readmode and args.itime == 1:
        args.itime = 11

    config = Config(
        mode=args.mode,
        readmode=args.readmode,
    )
    run(config, args)

if __name__ == "__main__":
    main()
