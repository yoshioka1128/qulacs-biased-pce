# src/app/postprocess_greedy_time_resolved_allzero.py

import os
import json
import csv
import traceback

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
from gurobi_energy_mathopt.data_loader import load_selected_originals


# =========================================================
# settings
# =========================================================

iseed = 42
method = "BFGS"
type_ansatz = "all2all"

# 実行対象
DUMMY_MODE = "nobias"

# time resolved 用
IT_START = 11
IT_END = 20
NT = 1


def build_norm_function(record, shift):
    Cmin, Cmax, frob_norm, _ = record["normalize"]

    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)

    return norm

def run_greedy_allzero_postprocess(
    nodes: int,
    rate: float,
    it: int,
):
    key = (nodes, rate, DUMMY_MODE)

    if key not in NODE_CONFIG:
        print(f"[skip] config not found: {key}")
        return None

    cfg = NODE_CONFIG[key]

    # =====================================================
    # 保存先 directory を取得
    # （既存の result file を使って target_dir を特定）
    # =====================================================
    try:
        target_dir, result_file = get_result_file_from_node_config(
            nodes=nodes,
            rate=rate,
            mode=DUMMY_MODE,
            method=method,
            iseed=iseed,
            type_ansatz=type_ansatz,
            it=it,
            nT=1,
            readmode=True,
        )
    except Exception as e:
        print(
            f"[skip] result file not found: "
            f"nodes={nodes}, it={it}, error={e}"
        )
        return None

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

    print("\n=== processing ===")
    print(f"nodes={nodes}")
    print(f"rate={rate}")
    print(f"it={it}")

    # =====================================================
    # Gurobiデータ読み込み
    # =====================================================
    df_selected = load_selected_originals(nodes, iseed)

    if df_selected is None:
        print(
            f"[skip] csv not found: "
            f"selected_originals_L{nodes}_iseed{iseed}.csv"
        )
        return None

    consumer_list = df_selected["Consumer"].tolist()

    # =====================================================
    # time resolved Hamiltonian
    # =====================================================
    frob_norm, shift, dJ, dhex = prepare_int_from_d(
        consumer_list,
        nodes,
        it,
        NT,
        rate,
    )

    # greedy は対称化した相互作用を使用
    dJ_sym = dJ + dJ.T

    # =====================================================
    # ★ all-1 初期解
    # =====================================================
    x = [1] * nodes

    # =====================================================
    # greedy post-processing
    # =====================================================
    x_local, cost_local = greedy_ising(
        dJ_sym,
        dhex,
        x,
    )

    norm = build_norm_function(record, shift)

    result_data = {
        "Calculated Minimum Energy [norm, raw]": [
            norm(cost_local),
            cost_local,
        ],
        "Solution for Minimum Energy": x_local,
    }

    output_path = os.path.join(
        target_dir,
        f"greedy_allzero_time_resolved_it{it}_results.json"
    )

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(
        f"saved -> {output_path}"
    )

    return {
        "nodes": nodes,
        "it": it,
        "cost": norm(cost_local),
    }


# =========================================================
# main
# =========================================================
def main():
    results_by_rate = defaultdict(list)

    for it in range(IT_START, IT_END + 1):
        for (nodes, rate, mode), _cfg in NODE_CONFIG.items():
            # ★ modeは無視
            if mode != DUMMY_MODE:
                continue

            try:
                result = run_greedy_allzero_postprocess(
                    nodes=nodes,
                    rate=rate,
                    it=it,
                )

                if result is not None:
                    results_by_rate[(mode, rate)].append(result)

            except Exception:
                print(
                    f"[error] nodes={nodes}, "
                    f"rate={rate}, mode={mode}, it={it}"
                )
                traceback.print_exc()

    # =====================================================
    # mode + rate ごとに CSV 保存
    # =====================================================
    for (mode, rate), rows in results_by_rate.items():
        csv_path = os.path.join(
            "outputs/power_opt/csv",
            f"greedy_time_resolved_summary_rate{rate}.csv"
        )

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")

            writer.writerow([
                "nodes",
                "it",
                "cost",
            ])

            for row in rows:
                writer.writerow([
                    row["nodes"],
                    row["it"],
                    row["cost"],
                ])

        print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
