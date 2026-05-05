import os
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import json
from src.config.full_config import build_config
from src.config.pipeline_config import PIPELINE_CONFIG

from src.core.spin_init import generate_spin
from src.core.utils import prepare_int_from_d
from src.core.optimizer import greedy_ising
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_result_row

# =========================================================
# settings
# =========================================================
it = 1
nT = 24
iseed = 42

DUMMY_BIAS_MODE = "nobias"

def build_norm_function(Cmin, Cmax, frob_norm, shift):
    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)

    return norm


def one_sample(nodes, dJ_sym, dhex, seed, norm):
    x0 = generate_spin("random", nodes, seed=seed)
    _, cost = greedy_ising(dJ_sym, dhex, x0)
    return norm(cost), x0

def run_greedy_random(nodes: int, rate: float, pipeline: str, nsample: int = 1000, iran: int = 42):

    key = (nodes, rate, pipeline)

    if key not in PIPELINE_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    cfg = build_config(nodes, rate, pipeline, DUMMY_BIAS_MODE)

    # ===== normalize =====
    Cmin, Cmax, frob_norm = load_gurobi_result_row(nT, nodes, rate, iseed, it)

    # ===== Ising =====
    df_selected = load_selected_originals(nodes, iseed)
    if df_selected is None:
        print("[skip] csv not found")
        return

    consumer_list = df_selected["Consumer"].tolist()

    frob_norm, shift, dJ, dhex = prepare_int_from_d(
        consumer_list,
        nodes,
        it,
        nT,
        rate,
    )

    dJ_sym = dJ + dJ.T
    norm = build_norm_function(Cmin, Cmax, frob_norm, shift)

    # ===== sampling =====
    results = [
        one_sample(nodes, dJ_sym, dhex, i + iran, norm)
        for i in range(nsample)
    ]

    eng = np.array([r[0] for r in results])
    x_ite = [r[1] for r in results]

    mineng = eng.min()
#    meaneng = eng.mean()
    meaneng = np.mean(eng)
    stdeng  = np.std(eng)

    # ===== 出力 =====
    output_dir = f"outputs/power_opt/greedy_optimized/nT{nT}_rate{rate}_{nodes}nodes"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- JSON（summary） ---
    data = {
        "Calculated Minimum Energy [norm, row, mean, row]": [
            float(mineng),
            float(meaneng),
        ],
        "Solution for Minimum Energy": x_ite[int(np.argmin(eng))].tolist(),
    }

    with open(
        f"{output_dir}/greedyran_nsample{nsample}_iran{iran}_time{it}_nT{nT}_rate{rate}_{nodes}nodes_iseed{iseed}.json",
        "w"
    ) as f:
        json.dump(data, f, indent=2)

    # --- CSV（最重要：legacy互換） ---
    csv_path = f"{output_dir}/sampling_greedyran_nsample{nsample}_iran{iran}_time{it}_nT{nT}_rate{rate}_{nodes}nodes_iseed{iseed}.csv"

    with open(csv_path, "w", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(eng)   # ← これが完全一致ポイント

    print(f"[done] nodes={nodes}, mean={meaneng:.4f}, min={mineng:.4f}")

    return {
        "nodes": nodes,
        "rate": rate,
        "mean": float(meaneng),
        "std": float(stdeng),
    }


def main():
    results = []

    for (nodes, rate, pipeline) in PIPELINE_CONFIG.keys():
        if pipeline != "averaged":
            continue
        print(nodes, rate, pipeline)

        try:
            result = run_greedy_random(nodes, rate, pipeline, nsample=1000)

            if result:
                results.append({
                    "nodes": nodes,
                    "rate": rate,
                    "cost": result["mean"],   # ← 平均
                    "std": result["std"],     # ← 追加
                })

        except Exception:
            print(f"[error] nodes={nodes}, rate={rate}, pipeline={pipeline}")
            traceback.print_exc()

    # ===== DataFrame化 =====
    df = pd.DataFrame(results)

    # 並び替え
    df = df.sort_values(["rate", "nodes"])

    # ===== CSV保存 =====
    csv_path = os.path.join(
        "outputs/power_opt/csv",
        "greedy_random_averaged_summary_all.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"saved -> {csv_path}")

if __name__ == "__main__":
    main()
