import glob
import re
import argparse
import os
import csv

# ----------------------
# 引数
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--rate", type=float, default=0.1, help="rate (default: 0.1)")
parser.add_argument("--topk", type=int, default=5, help="number of outputs (default: 5)")
args = parser.parse_args()

rate = args.rate
topk_num = args.topk

# ----------------------
configs = {
    18:  "18nodes_4qubits_2body",
    60:  "60nodes_6qubits_3body",
    210: "210nodes_8qubits_4body",
    756: "756nodes_10qubits_5body",
    2772: "2772nodes_12qubits_6body",
}

save_dir = "scripts/data/results"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/best_rate{rate}_top{topk_num}.csv"


def make_base(m, desc):
    return f"outputs/power_opt/time1_nT24_rate{rate}_{desc}_ninit5_depth5_all2all_methodBFGS_iseed42"

pattern = re.compile(
    r"alphasc(?P<alpha>[-\d\.]+)_beta(?P<beta>[-\d\.]+)_init(?P<init>\d+)"
)

def extract_value(filepath):
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == 2:
                return float(line.split()[0].replace(",", ""))
    return None

def parse_filename(filepath):
    m = pattern.search(filepath)
    if not m:
        return None
    return {
        "alphasc": float(m.group("alpha")),
        "beta": float(m.group("beta")),
        "init": int(m.group("init")),
        "bias": "True" if "bias" in filepath else "False",
        "reg_type": (
            "typex" if "typex" in filepath else
            "typey" if "typey" in filepath else
            "-"
        )
    }

def find_topk(files, k=5):
    results = []
    for fpath in files:
        val = extract_value(fpath)
        if val is None:
            continue

        info = parse_filename(fpath)
        if info is None:
            continue

        results.append({"value": val, **info})

    results.sort(key=lambda x: x["value"])
    return results[:k]

# ----------------------
# 実行
# ----------------------
print("m, bias, alphasc, beta, init, reg_type, value")

with open(save_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow([
        "m", "bias", "alphasc",
        "beta", "init", "reg_type", "value"
    ])
    for m, desc in configs.items():
        base = make_base(m, desc)

        # 非bias
        files = glob.glob(f"{base}/results_backprop_alphasc*.json")
        topk = find_topk(files, k=topk_num)
        for r in topk:
            writer.writerow([
                m,
                r["bias"],
                r["alphasc"],
                r["beta"],
                r["init"],
                r["reg_type"],
                r["value"]
            ])
            print(f"{m}, {r['bias']}, {r['alphasc']}, {r['beta']}, {r['init']}, {r['reg_type']}, {r['value']}")
        print()

        # bias
        files = glob.glob(f"{base}_bias/results_backprop_bias_alphasc*.json")
        topk = find_topk(files, k=topk_num)
        for r in topk:
            writer.writerow([
                m,
                r["bias"],
                r["alphasc"],
                r["beta"],
                r["init"],
                r["reg_type"],
                r["value"]
            ])
            print(f"{m}, {r['bias']}, {r['alphasc']}, {r['beta']}, {r['init']}, {r['reg_type']}, {r['value']}")
        print()
