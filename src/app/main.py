# main.py
import argparse
from src.app.runner import run
from src.config.config import Config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m", type=int, default=60)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--itime", type=int, default=1)
    parser.add_argument("--nT", type=int, default=24)
    parser.add_argument("--rate", type=float, default=0.1)

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
    config = Config()
    config.normalize(args.bias)
    run(config, args)

if __name__ == "__main__":
    main()
