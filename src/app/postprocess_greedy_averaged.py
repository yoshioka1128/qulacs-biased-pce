# src/app/postprocess_greedy.py
import os
import json
import re
import csv
import traceback
import pandas as pd

from src.config.full_config import build_config
from src.config.model_config import MODEL_CONFIG
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
BIAS_MODES = ["nobias", "bias_x", "bias_y"]

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
    model: str,
    bias_mode: str
):
    key = (nodes, rate, bias_mode)

    if key not in MODEL_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    cfg = build_config(nodes, rate, model, bias_mode)

    try:
        _, result_file = get_result_file_from_node_config(
            cfg,
            nodes=nodes,
            rate=rate,
            model=model,
            bias_mode=bias_mode,
            method=method,
            iseed=iseed,
            type_ansatz=type_ansatz,
        )
    except Exception as e:
        print(
            f"[skip] result file not found: "
            f"nodes={nodes}, rate={rate}, bias_mode={bias_mode}, error={e}"
        )
        return

    print("\n=== processing ===")
    print(f"nodes={nodes}")
    print(f"rate={rate}")
    print(f"bias_mode={bias_mode}")
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
        "bias_mode": bias_mode,
    }

# =========================================================
# main
# =========================================================
def main():
    model = "averaged"

    results = []

    for (nodes, rate, bias_mode) in MODEL_CONFIG.keys():
        if bias_mode not in BIAS_MODES:
            continue

        try:
            result = run_greedy_postprocess(
                nodes=nodes,
                rate=rate,
                model=model,
                bias_mode=bias_mode,
            )

            results.append({
                "nodes": nodes,
                "rate": rate,
                "bias_mode": bias_mode,
                "cost_wo_pp": result["cost_wo_pp"],
                "cost": result["cost"],
                "loss": result["loss"],
            })

        except Exception:
            print(
                f"[error] nodes={nodes}, "
                f"rate={rate}, model={model}, bias_mode={bias_mode}"
            )
            traceback.print_exc()

    # ===== DataFrame化 =====
    df = pd.DataFrame(results)

    # 空チェック（安全対策）
    if df.empty:
        print("No data to save.")
        return

    # （おすすめ）並び替え
    df = df.sort_values(["rate", "bias_mode", "nodes"])

    # （任意）列順を明示（崩れ防止）
    df = df[[
        "nodes",
        "rate",
        "bias_mode",
        "cost_wo_pp",
        "cost",
        "loss",
    ]]

    # ===== CSV保存 =====
    csv_path = os.path.join(
        "outputs/power_opt/csv",
        "pce_greedy_averaged_summary_all.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"saved -> {csv_path}")
    
if __name__ == "__main__":
    main()
