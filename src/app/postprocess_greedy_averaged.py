# src/app/postprocess_greedy.py
import os
import json
import re
import csv

import pandas as pd

from collections import defaultdict
from src.config.node_config import NODE_CONFIG
from src.analysis.loader import (
    get_result_file_from_node_config,
    load_result_json,
    build_result_record,
)
from src.analysis.parser import parse_result_filename
from src.core.utils import prepare_int_from_d
from src.core.optimizer import greedy_ising
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_results

# =========================================================
# settings
# =========================================================

iseed = 42
method = "BFGS"
type_ansatz = "all2all"

# 実行対象
MODES = ["nobias", "bias_x", "bias_y"]

# =========================================================
# helpers
# =========================================================

def build_norm_function(record, shift):
    """
    legacyコードの normalize をそのまま利用
    """

    Cmin, Cmax, frob_norm, _ = record["normalize"]

    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)

    return norm


def run_greedy_postprocess(
    nodes: int,
    rate: float,
    mode: str,
):
    key = (nodes, rate, mode)

    if key not in NODE_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    cfg = NODE_CONFIG[key]

    # =====================================================
    # NODE_CONFIG から result file を一意に取得
    # =====================================================
    try:
        target_dir, result_file = get_result_file_from_node_config(
            nodes=nodes,
            rate=rate,
            mode=mode,
            method=method,
            iseed=iseed,
            type_ansatz=type_ansatz,
        )
    except Exception as e:
        print(
            f"[skip] result file not found: "
            f"nodes={nodes}, mode={mode}, error={e}"
        )
        return

    print("\n=== processing ===")
    print(f"nodes={nodes}")
    print(f"rate={rate}")
    print(f"mode={mode}")
    print(f"result_file={result_file}")

    # =====================================================
    # quantum result 読み込み
    # =====================================================
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
    # Gurobiデータ読み込み
    # =====================================================
    df_selected = load_selected_originals(nodes, iseed)
    if df_selected is None:
        print(f"[skip] csv not found: selected_originals_L{nodes}_iseed{iseed}.csv")
        return
    consumer_list = df_selected["Consumer"].tolist()

    # legacy と同じ
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
    # 量子計算の解
    # =====================================================
    x = [int(v) for v in record["solution"]]

    # greedy は対称化した相互作用を使用
    dJ_sym = dJ + dJ.T

    # =====================================================
    # greedy post-processing
    # =====================================================
    x_local1, cost_local1 = greedy_ising(
        dJ_sym,
        dhex,
        x,
    )

    norm = build_norm_function(record, shift)

    data = {
        "Calculated Minimum Energy [norm, row]": [
            norm(cost_local1),
            cost_local1,
        ],
        "Solution for Minimum Energy": x_local1,
    }

    print(nodes, record['cost w/o pp'], norm(cost_local1), norm(record['loss']))

    return {
        "nodes": nodes,
        "cost_wo_pp": record['cost w/o pp'],
        "cost": norm(cost_local1),
        "loss": norm(record['loss']),
        "mode": mode,
    }

# =========================================================
# main
# =========================================================
import traceback


def main():
    results_by_mode_rate = defaultdict(list)
    for (nodes, rate, mode), _cfg in NODE_CONFIG.items():
        if mode not in MODES:
            continue

        try:
            result = run_greedy_postprocess(
                nodes=nodes,
                rate=rate,
                mode=mode,
            )
            results_by_mode_rate[(mode, rate)].append(result)

        except Exception as e:
            print(
                f"[error] nodes={nodes}, "
                f"rate={rate}, mode={mode}"
            )

            traceback.print_exc()

    # ===== mode + rate ごとに最後にCSV保存 =====
    for (mode, rate), rows in results_by_mode_rate.items():

        csv_path = os.path.join(
            "outputs/power_opt",
            f"pce_greedy_averaged_summary_rate{rate}_{mode}.csv"
        )

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "nodes",
                "cost_wo_pp",
                "cost",
                "loss"
            ])

            for row in rows:
                writer.writerow([
                    row["nodes"],
                    row["cost_wo_pp"],
                    row["cost"],
                    row["loss"]
                ])

        print(f"saved -> {csv_path}")
    
if __name__ == "__main__":
    main()
