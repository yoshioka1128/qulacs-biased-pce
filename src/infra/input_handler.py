import os, csv
import numpy as np

def get_user_input():
    n_qubits = int(input("Number of qubits (6): ") or 6)
    k = int(input("Interaction order k (3): ") or 3)
    m = int(input("Number of nodes (60): ") or 60)
    depth0 = int(m/(n_qubits*(n_qubits-1)/2.0 + n_qubits*2.0)) # m ~ nparam
    depth = int(input("Circuit depth (5): ") or max(5, depth0))
    alphasc = float(input("alphasc (e.g., 1.5, coeffient for n_qubits**np.floor(k/2)): ") or 0.1)
    beta = float(input("beta (0.0): ") or 0.1)
    type_ansatz = str(input("Type ansatz (all2all): ").strip() or 'all2all')
    itime = int(input("Time of DR request (1): ") or 1)
    nT = int(input("Time step (24): ") or 24)
    rate = float(input("procurement rate (0.25): ") or 0.25)
    print('depth: ', depth)

    return itime, nT, rate, n_qubits, k, m, depth, alphasc, beta, type_ansatz

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
