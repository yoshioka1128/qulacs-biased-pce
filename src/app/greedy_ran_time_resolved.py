# src/app/greedy_ran_time_resolved.py

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

from gurobi_energy_mathopt.data_loader import (
    load_selected_originals,
    load_gurobi_result_row,
)

from src.domain.power.data_loader import PowerDataLoader
from src.analysis.aggregator import evaluate_solution
from joblib import Parallel, delayed

# =========================================================
# settings
# =========================================================
rate0 = float(input('rate (0.1, 0.4): ') or 0.1)
nsample0 = int(input('nsample (1000): ') or 1000)

nT = 1
iseed = 42
DUMMY_BIAS_MODE = "nobias"

# time resolved 用
IT_START = 11
IT_END = 20
NT = 1

def build_norm_function(Cmin, Cmax, frob_norm, shift):
    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)
    return norm


def one_sample(nodes, dJ_sym, dhex, seed, norm):
    x0 = generate_spin("random", nodes, seed=seed)
    x0, cost = greedy_ising(dJ_sym, dhex, x0)
    return norm(cost), x0


def run_greedy_random_time_resolved(
    nodes: int,
    rate: float,
    pipeline: str,
    nsample: int = 1000,
    iran: int = 42,
):

    key = (nodes, rate, pipeline)
    if key not in PIPELINE_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    # ===== DataLoader =====
    loader = PowerDataLoader(nodes, iseed, rate)

    # ===== consumer =====
    df_selected = load_selected_originals(nodes, iseed)
    if df_selected is None:
        print("[skip] csv not found")
        return

    consumer_list = df_selected["Consumer"].tolist()

    results_per_hour = []

    # ===== 時間ループ =====
    for it in range(IT_START, IT_END + 1):

        print(f"[run] nodes={nodes}, rate={rate}, hour={it}")

        # ===== normalize =====
        Cmin, Cmax, frob_norm = load_gurobi_result_row(
            nT, nodes, rate, iseed, it
        )

        # ===== Ising =====
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
        results = []
        for i in range(nsample):
            print('sampling', i)
            results.append(one_sample(nodes, dJ_sym, dhex, i + iran, norm))

#        results = [
#            one_sample(nodes, dJ_sym, dhex, i + iran, norm)
#            for i in range(nsample)
#        ]
#        results = Parallel(n_jobs=25, backend="loky")(
#            delayed(one_sample)(nodes, dJ_sym, dhex, i + iran, norm)
#            for i in range(nsample)
#        )

        eng = np.array([r[0] for r in results])
        x_ite = [r[1] for r in results]

        # ===== energy stats =====
        mineng = eng.min()
        meaneng = np.mean(eng)
        stdeng = np.std(eng)

        # ===== evaluate (dev / var / cv) =====
        var_list = []
        dev_list = []
        cv_list = []

        for i, sol in enumerate(x_ite):
            print('evaluate:', i)
            var, dev, Ptot = evaluate_solution(it, sol, loader)

            cv_list.append(np.sqrt(var)/Ptot)
            dev_list.append(dev)

#            if Ptot != 0:
#                cv_list.append(np.sqrt(var) / Ptot)

#        var_mean = float(np.mean(var_list))
#        var_std = float(np.std(var_list))

        dev_mean = float(np.mean(dev_list)) # deviation
        dev_std = float(np.std(dev_list))

        cv_mean = float(np.mean(cv_list)) if len(cv_list) > 0 else 0.0
        cv_std = float(np.std(cv_list)) if len(cv_list) > 0 else 0.0

        # ===== 保存 =====
        output_dir = f"outputs/power_opt/greedy_optimized/nT{nT}_rate{rate}_{nodes}nodes"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # --- CSV (energy sampling) ---
        csv_path = f"{output_dir}/sampling_greedyran_nsample{nsample}_iran{iran}_time{it}_nT{nT}_rate{rate}_{nodes}nodes_iseed{iseed}.csv"
        with open(csv_path, "w", newline="\n") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(eng)

        # --- JSON summary ---
        data = {
            "hour": it,
            "energy": {
                "min": float(mineng),
                "mean": float(meaneng),
                "std": float(stdeng),
            },
            "dev": {
                "mean": dev_mean,
                "std": dev_std,
            },
#            "var": {
#                "mean": var_mean,
#                "std": var_std,
#            },
            "cv": {
                "mean": cv_mean,
                "std": cv_std,
            },
            "best_solution": x_ite[int(np.argmin(eng))],
        }
        print(it, dev_mean,"(", dev_std, ")", cv_mean, "(", cv_std, ")")

        with open(
            f"{output_dir}/greedyran_time{it}_nsample{nsample}_iran{iran}_rate{rate}_{nodes}nodes.json",
            "w",
        ) as f:
            json.dump(data, f, indent=2)

        results_per_hour.append({
            "hour": it,
            "dev_mean": dev_mean,
            "dev_std": dev_std,
#            "var_mean": var_mean,
#            "var_std": var_std,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cost_mean": meaneng,
            "cost_std": stdeng,
        })

    return results_per_hour


def main():
    all_results = []

    for (nodes, rate, pipeline) in PIPELINE_CONFIG.keys():
        if nodes != 756: continue
        if pipeline != "time_resolved": continue
        if rate != rate0: continue

        print(nodes, rate, pipeline)

        try:
            results = run_greedy_random_time_resolved(
                nodes, rate, pipeline, nsample=nsample0
            )

            if results:
                for r in results:
                    r["nodes"] = nodes
                    r["rate"] = rate
                    all_results.append(r)

        except Exception:
            print(f"[error] nodes={nodes}, rate={rate}, pipeline={pipeline}")
            traceback.print_exc()

    df = pd.DataFrame(all_results)
    df = df.sort_values(["rate", "nodes", "hour"])

    csv_path = os.path.join(
        "outputs/power_opt/csv",
        f"greedy_random_time_resolved_summary_all_rate{rate0}_nsample{nsample0}.csv"
    )

    df.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
