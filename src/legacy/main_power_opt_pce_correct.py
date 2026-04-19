from core.input_hundler import get_user_input, setup_output_dirs
from core.graph_handler import prepare_int
from core.ansatz_factory import select_ansatz
from core.optimizer import read_optimize_fast
from core.result_handler import save_results, save_results_fast
from pce import pauli_correlation_encode, show_observable
from graph_utils import visualize_graph
from utils import spin_to_number
import numpy as np
import pandas as pd
import csv, datetime, os, json
graph_type = "power_opt"
# 実行モード選択
choice = str(input("USE_NEW モードで実行しますか？ (y/n):(y) ").strip().lower() or "y")
USE_NEW = (choice == "y")
mode_str = "new" if USE_NEW else "old"

# learning
choice = str(input("学習モードで実行しますか？ (y/n):(n) ").strip().lower() or "n")
LEARN = (choice == "y")
if LEARN:
    nprob = int(input("Number of problems (250): ") or 250)
    ninit = int(input("Number of initialization (5): ") or 5)
    iseed = int(input("seed for initial angles (timestamp): ") or int(datetime.datetime.now().timestamp()))
else:
    nprob = 1 # 問題数
    ninit = int(input("Number of initialization (5): ") or 5)
    iseed = int(input("seed for initial angles (42): ") or 42)
rng = np.random.default_rng(iseed) # 乱数の種固定

# method & verbose
method = str(input("optimization method (BFGS): ").strip() or 'BFGS')
verbose = int(input("verbose 0 or 1 (1): ") or 1)

def main():
    # Get user-defined parameters
    it, nT, rate, n_qubits, k, m, depth, alphasc, beta, type_ansatz = get_user_input()
    if LEARN:
        alphascs = [0.01, 0.1, 0.5, 1.0, 1.5]
        betas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    else:
        alphascs, betas = [alphasc], [beta]

    # Set up output directory for results and figures
    output_dir, _ = setup_output_dirs(LEARN, mode_str, nprob, ninit, it, nT, rate, m, type_ansatz, n_qubits, k, depth, method, iseed, alphasc, beta)

    # Select and build the ansatz
    ansatz = select_ansatz(type_ansatz, n_qubits, depth)
    if verbose: print(ansatz.get_qulacs_circuit())

    # application of backprop
    choice = str(input("backprop ? y/n (n):").strip() or "n")
    USE_BACKPROP = (choice == "y")

    # read file
    readmode = str(input("read mode ? y/n (n):").strip() or "n")
    if readmode == "n":
        ninit2=ninit
    else:
        ninit2=1
        iinit = int(input("read initialization number (0): ") or 0)
        choice = str(input("read progress y/n (y):") or "y")
        if choice == "y":
            strread="progress"
            strparam = "parameters"
        else:
            strread="results"
            strparam = "Minimum Parameters"
        print('read', strread)
        alpha = alphasc * n_qubits ** np.floor(k / 2)
        read_file = os.path.join(output_dir, f"{strread}_alpha{alpha}_beta{beta}_init{iinit}_iseed{iseed}.json")
        with open(read_file, "r") as f:
            data = json.load(f)
        init_para = np.array(data[f"{strparam}"])
        output_dir = output_dir + "read"
        os.makedirs(output_dir, exist_ok=True)

    # maximum number of iteration
    maxiter = int(input("maxiter (10000): ") or 10000)

    for iprob in range(nprob):
        # Prepare graph and compute related constants
        dJ, dhex, Cmin, Cmax, frob_norm, shift, consumer_list = prepare_int(graph_type, m, it, nT, rate, rng, USE_NEW, iseed)
        if not LEARN:
            df_selected_originals = pd.DataFrame(consumer_list, columns=['Consumer'])
            df_selected_originals.to_csv(f'{output_dir}/consumer_list_iseed{iseed}.csv', index=False)

        # Embed bitstrings into correlators using Pauli Correlation Encoding
        pce = pauli_correlation_encode(m, n_qubits, k)
        if verbose: show_observable(pce)

        # Run optimization
        for alphasc in alphascs:
            alpha = alphasc * n_qubits ** np.floor(k / 2)
            for beta in betas:
                for iinit in range(ninit2):
                    if readmode == "n": init_para = rng.uniform(0, 2.0*np.pi, size=ansatz.get_parameter_count())
                    result, history, elapsed_time = read_optimize_fast(init_para, method, dJ, dhex, n_qubits, ansatz, pce,
                                                                       alpha, beta, verbose, Cmin, Cmax,
                                                                       frob_norm, shift, iinit, iseed, output_dir,
                                                                       USE_BACKPROP, maxiter)
                    mineng, minnum,  = save_results_fast(output_dir, result, history, dJ, dhex, ansatz, pce, n_qubits,
                                                         Cmin, Cmax, frob_norm, shift, m,
                                                         alpha, beta, elapsed_time, iinit, iseed, verbose, LEARN, USE_BACKPROP)
                    # CSV出力
                    print(alphasc, beta, iprob, iinit, mineng, minnum)
#                    with open(f'{output_dir}/{csvfile}', mode='a', newline='', encoding="utf-8") as f:
#                        writer = csv.writer(f, lineterminator="\n")
#                        writer.writerow([alphasc, beta, iprob, iinit, mineng, minnum])

if __name__ == "__main__":
    main()
