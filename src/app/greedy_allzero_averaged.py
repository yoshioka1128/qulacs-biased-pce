# src/app/postprocess_greedy_allzero.py

import os
import traceback
import pandas as pd
from collections import defaultdict
from src.config.full_config import build_config
from src.config.pipeline_config import PIPELINE_CONFIG

from src.analysis.loader import (
    get_result_file_from_node_config,
    load_result_json,
    build_result_record,
)
from src.core.spin_init import generate_spin
from src.analysis.parser import parse_result_filename
from src.core.utils import prepare_int_from_d
from src.core.optimizer import greedy_ising
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_result_row


# =========================================================
# settings
# =========================================================
it = 1
nT = 24
iseed = 42
method = "BFGS"
type_ansatz = "all2all"

# ★ 固定（normalize取得用）
DUMMY_BIAS_MODE = "nobias"

def build_norm_function(Cmin, Cmax, frob_norm, shift):
    def norm(raw_cost):
        return (raw_cost * frob_norm + shift - Cmin) / (Cmax - Cmin)

    return norm


def run_greedy_allzero(nodes: int, rate: float, model: str):
    key = (nodes, rate, model)

    if key not in PIPELINE_CONFIG:
        print(f"[skip] config not found: {key}")
        return

    cfg = build_config(nodes, rate, model, DUMMY_BIAS_MODE)

    # =====================================================
    # normalize取得（量子結果は使わない）
    # =====================================================
    Cmin, Cmax, frob_norm = load_gurobi_result_row(nT, nodes, rate, iseed, it)

#    try:
#        _, result_file = get_result_file_from_node_config(
#            cfg,
#            nodes=nodes,
#            rate=rate,
#            model=model,
#            bias_mode=DUMMY_BIAS_MODE,
#            method=method,
#            iseed=iseed,
#            type_ansatz=type_ansatz,
#        )
#    except Exception as e:
#        print(
#            f"[skip] result file not found:"
#            f"nodes={nodes}, rate={rate}, model:{model}, error={e}"
#        )
#        return
#
#    meta = parse_result_filename(result_file)
#    data = load_result_json(result_file)
#    record = build_result_record(
#        meta=meta,
#        data=data,
#        nodes=nodes,
#        qubits=cfg.n_qubits,
#        body=cfg.k,
#        best_file=result_file,
#    )

    # =====================================================
    # Ising構築
    # =====================================================
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

    # =====================================================
    # ★ all-1 初期解
    # =====================================================
    x = generate_spin('allzero', nodes)

    dJ_sym = dJ + dJ.T

    x_local, cost_local = greedy_ising(
        dJ_sym,
        dhex,
        x,
    )

#    norm = build_norm_function(record, shift)
    norm = build_norm_function(Cmin, Cmax, frob_norm, shift)

    return {
        "nodes": nodes,
        "rate": rate,
        "cost": norm(cost_local),
    }


# =========================================================
def main():
    results = []

    for (nodes, rate, model) in PIPELINE_CONFIG.keys():
        if model != "averaged":
            continue

        print(nodes, rate, model)

        try:
            result = run_greedy_allzero(nodes, rate, model)

            if result:
                results.append({
                    "nodes": nodes,
                    "rate": rate,
                    "cost": result["cost"],
                })

        except Exception:
            print(f"[error] nodes={nodes}, rate={rate}, model={model}")
            traceback.print_exc()

    # ===== DataFrame化 =====
    df = pd.DataFrame(results)

    # （任意）並び替え：見やすく＆後処理しやすい
    df = df.sort_values(["rate", "nodes"])

    # ===== CSV保存 =====
    csv_path = os.path.join(
        "outputs/power_opt/csv",
        "greedy_allzero_averaged_summary_all.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
