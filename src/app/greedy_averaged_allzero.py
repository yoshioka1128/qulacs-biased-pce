# src/app/postprocess_greedy_allzero.py

import os
import csv
from collections import defaultdict
import traceback

from src.config.node_config import NODE_CONFIG
from src.analysis.loader import (
    get_result_file_from_node_config,
    load_result_json,
    build_result_record,
)
from src.analysis.parser import parse_result_filename
from src.core.utils import prepare_int_from_d
from src.core.optimizer import greedy_ising
from gurobi_energy_mathopt.data_loader import load_selected_originals


# =========================================================
# settings
# =========================================================
iseed = 42
method = "BFGS"
type_ansatz = "all2all"

# ★ 固定（normalize取得用）
DUMMY_MODE = "nobias"


# =========================================================
def build_norm_function(record, shift):
    Cmin, Cmax, frob_norm, _ = record["normalize"]

    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)

    return norm


def run_greedy_all1(nodes: int, rate: float):
    key = (nodes, rate, DUMMY_MODE)

    if key not in NODE_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    cfg = NODE_CONFIG[key]

    # =====================================================
    # normalize取得（量子結果は使わない）
    # =====================================================
    try:
        _, result_file = get_result_file_from_node_config(
            nodes=nodes,
            rate=rate,
            mode=DUMMY_MODE,
            method=method,
            iseed=iseed,
            type_ansatz=type_ansatz,
        )
    except Exception:
        print(f"[skip] result file not found: nodes={nodes}")
        return

    meta = parse_result_filename(result_file)
    data = load_result_json(result_file)
    record = build_result_record(
        meta=meta,
        data=data,
        nodes=nodes,
        qubits=cfg.n_qubits,
        body=cfg.k,
        best_file=result_file,
    )

    # =====================================================
    # Ising構築
    # =====================================================
    df_selected = load_selected_originals(nodes, iseed)
    if df_selected is None:
        print("[skip] csv not found")
        return

    consumer_list = df_selected["Consumer"].tolist()

    it = 1
    nT = 24

    frob_norm, shift, dJ, dhex = prepare_int_from_d(
        consumer_list,
        nodes,
        it,
        nT,
        rate,
    )

    # =====================================================
    # ★ all-1 初期解
    # =====================================================
    x = [1] * nodes

    dJ_sym = dJ + dJ.T

    x_local, cost_local = greedy_ising(
        dJ_sym,
        dhex,
        x,
    )

    norm = build_norm_function(record, shift)

    return {
        "nodes": nodes,
        "rate": rate,
        "cost": norm(cost_local),
    }


# =========================================================
def main():
    results_by_rate = defaultdict(list)

    for (nodes, rate, mode), _cfg in NODE_CONFIG.items():
        # ★ modeは無視
        if mode != DUMMY_MODE:
            continue

        try:
            result = run_greedy_all1(nodes, rate)
            if result:
                results_by_rate[rate].append(result)

        except Exception:
            print(f"[error] nodes={nodes}, rate={rate}")
            traceback.print_exc()

    # CSV保存
    for rate, rows in results_by_rate.items():
        csv_path = os.path.join(
            "outputs/power_opt/csv",
            f"greedy_allzero_averaged_summary_rate{rate}.csv"
        )

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")

            writer.writerow([
                "nodes",
                "rate",
                "cost",
            ])

            for row in rows:
                writer.writerow([
                    row["nodes"],
                    row["rate"],
                    row["cost"],
                ])

        print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
