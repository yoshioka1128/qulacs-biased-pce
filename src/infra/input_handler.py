# input_handler.py

import os, csv

def setup_output_dirs(config, mode_str, it, nT, rate, m, type_ansatz, n_qubits, k, depth):
    LEARN = config.learn
    nprob = config.nprob
    ninit = config.ninit
    method = config.method
    iseed = config.iseed
    bias_mode = config.bias_mode
    
    if LEARN:
        output_dir = f'outputs_learning/time{it}_nT{nT}_rate{rate}_nprob{nprob}_ninit{ninit}_{type_ansatz}_method{method}_iseed{iseed}/'
        os.makedirs(output_dir, exist_ok=True)
        csvfile = f'learning_{m}nodes_{n_qubits}qubits_{k}body_depth{depth}_{bias_mode}_iseed{iseed}.csv'
        csvpath = os.path.join(output_dir, csvfile)
        file_exists = os.path.isfile(csvpath)
        with open(csvpath, mode='a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            if not file_exists:
                writer.writerow(['alphasc', 'beta', 'problem', 'initialization', 'minimum energy', 'decimal number'])
    else:
        output_dir = f'outputs/power_opt/time{it}_nT{nT}_rate{rate}_{m}nodes_{n_qubits}qubits_{k}body_ninit{ninit}_depth{depth}_{type_ansatz}_method{method}_iseed{iseed}/'
        os.makedirs(output_dir, exist_ok=True)
    return output_dir
