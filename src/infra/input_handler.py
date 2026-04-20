import os, csv
import numpy as np
from src.config.node_config import NODE_CONFIG

def get_user_input():
    # --- m入力 ---
    m = int(input("Number of nodes (60): ") or 60)

    if m not in NODE_CONFIG:
        raise ValueError(f"Unsupported m={m}. Available: {list(NODE_CONFIG.keys())}")

    cfg = NODE_CONFIG[m]

    # --- NODE_CONFIGから取得（デフォルト） ---
    n_qubits_default = cfg.n_qubits
    k_default = cfg.k
    depth_default = cfg.depth
    alphasc_default = cfg.alphasc
    beta_default = cfg.beta

    # --- 上書き可能にする ---
    n_qubits = int(input(f"Number of qubits ({n_qubits_default}): ") or n_qubits_default)
    k = int(input(f"Interaction order k ({k_default}): ") or k_default)

    # depthはm, n_qubitsに依存するので再計算も可能
    depth0 = int(m / (n_qubits*(n_qubits-1)/2.0 + n_qubits*2.0))
    depth = int(input(f"Circuit depth ({depth_default}): ") or max(depth_default, depth0))

    alphasc = float(input(f"alphasc ({alphasc_default}): ") or alphasc_default)
    beta = float(input(f"beta ({beta_default}): ") or beta_default)

    # --- その他 ---
    type_ansatz = str(input("Type ansatz (all2all): ").strip() or 'all2all')
    itime = int(input("Time of DR request (1): ") or 1)
    nT = int(input("Time step (24): ") or 24)
    rate = float(input("procurement rate (0.25): ") or 0.25)

    return {
        "itime": itime,
        "nT": nT,
        "rate": rate,
        "n_qubits": n_qubits,
        "k": k,
        "m": m,
        "depth": depth,
        "alphasc": alphasc,
        "beta": beta,
        "type_ansatz": type_ansatz,
    }

def setup_output_dirs(LEARN, mode_str, nprob, ninit, it, nT, rate, m, type_ansatz, n_qubits, k, depth, method, iseed):
    if LEARN:
        output_dir = f'outputs_learning/time{it}_nT{nT}_rate{rate}_nprob{nprob}_ninit{ninit}_{type_ansatz}_method{method}_42seed_{mode_str}/'
        os.makedirs(output_dir, exist_ok=True)
        csvfile = f'learning_{m}nodes_{n_qubits}qubits_{k}body_depth{depth}_iseed{iseed}.csv'
        csvpath = os.path.join(output_dir, csvfile)
        file_exists = os.path.isfile(csvpath)
        with open(csvpath, mode='a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            if not file_exists:
                writer.writerow(['alphasc', 'beta', 'problem', 'initialization', 'minimum energy', 'decimal number'])
    else:
        output_dir = f'outputs/power_opt/time{it}_nT{nT}_rate{rate}_{m}nodes_{n_qubits}qubits_{k}body_ninit{ninit}_depth{depth}_{type_ansatz}_method{method}_iseed{iseed}_{mode_str}/'
        os.makedirs(output_dir, exist_ok=True)
    return output_dir
