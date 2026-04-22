# runner.py
import numpy as np
import pandas as pd
import os, json, itertools
import json

from src.core.graph_handler import prepare_int
from src.core.ansatz_factory import select_ansatz
from src.core.init_strategy import sample_init
from src.infra.input_handler import get_user_input, setup_output_dirs
from src.infra.result_handler import save_results_fast
from pce import pauli_correlation_encode, show_observable
from src.core.optimizer import read_optimize_fast
from src.config.node_config import NODE_CONFIG
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_results

def run(config, args):
    cfg = NODE_CONFIG[args.m]

    # --- デフォルト（NODE_CONFIG） ---
    params = {
        "m": args.m,
        "n_qubits": args.n_qubits or cfg.n_qubits,
        "k": args.k or cfg.k,
        "depth": args.depth or cfg.depth,
        "type_ansatz": args.type_ansatz,
        "itime": args.itime,
        "nT": args.nT,
        "rate": args.rate,
        "bias": args.bias
    }

    if args.batch:
        run_batch(config, params)
    else:
        alphasc = args.alphasc if args.alphasc is not None else cfg.alphasc
        beta = args.beta if args.beta is not None else cfg.beta
        run_single(config, params, alphasc, beta)

def run_batch(config, params):
#    beta_list = [0.5, 1.0, 1.5, 2.0]
    beta_list = [-0.1, 0.0, 0.1, 0.2]
#    alpha_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    alpha_list = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for beta, alphasc in itertools.product(beta_list, alpha_list):
        print(f"\n=== beta={beta}, alphasc={alphasc} ===")

        run_single(config, params, alphasc, beta)

def run_single(config, params, alphasc, beta):
    n_qubits = params["n_qubits"]
    depth = params["depth"]
    type_ansatz = params["type_ansatz"]

    ansatz = select_ansatz(type_ansatz, n_qubits, depth)

    rng = np.random.default_rng(config.iseed)
    it = params['itime']
    nT = params['nT']
    rate = params['rate']
    n_qubits = params["n_qubits"]
    k = params['k']
    m = params['m']
    depth = params["depth"]
    type_ansatz = params["type_ansatz"]
    USE_BIAS = params["bias"]
    
    if config.learn:
        alphascs = [0.01, 0.1, 0.5, 1.0, 1.5]
        betas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    else:
        alphascs, betas = [alphasc], [beta]

    output_dir = setup_output_dirs(
        config.learn, "new" if config.use_new else "old",
        config.nprob, config.ninit, it, nT, rate, m,
        type_ansatz, n_qubits, k, depth,
        config.method, config.iseed
    )

    ansatz = select_ansatz(type_ansatz, n_qubits, depth)

    if config.verbose:
        print(ansatz.get_qulacs_circuit())

    # read mode
    init_para = None
    ninit2 = config.ninit

    if config.readmode:
        ninit2 = 1
        iinit = 0

        read_file = os.path.join(
            output_dir,
            f"progress_alphasc{alphasc}_beta{beta}_init{iinit}_iseed{config.iseed}.json"
        )

        with open(read_file, "r") as f:
            data = json.load(f)

        init_para = np.array(data["parameters"])

        output_dir = os.path.join(output_dir, "read")
        os.makedirs(output_dir, exist_ok=True)

    for iprob in range(config.nprob):

        # load consumer_list
        df_selected_originals = load_selected_originals(m, config.iseed)
        consumer_list = df_selected_originals["Consumer"].tolist()
        # load results_list
        gurobi_result = load_gurobi_results(nT, m, rate, config.iseed)
#        row = df.loc[df["hour"] == it] # bug fix                                                                                                                       
#        Cmin, Cmax, frob_norm = row["obj_val_min"].iloc[0], row["obj_val_max"].iloc[0], row["frobenius_norm"].iloc[0]
        
        dJ, dhex, Cmin, Cmax, frob_norm, shift, consumer_list = prepare_int(
            "power_opt", m, it, nT, rate,
            rng, config.use_new, config.iseed,
            consumer_list=consumer_list,
            gurobi_result=gurobi_result,
        )

#        df = load_gurobi_results(nT, m, rate, iseed)
#        row = df.loc[df["hour"] == it]
#        Cmin = row["obj_val_min"].iloc[0]
#        Cmax = row["obj_val_max"].iloc[0]
#        frob_norm = row["frobenius_norm"].iloc[0]

        if not config.learn:
#            df = load_selected_originals(m, iseed)
#            df_selected_originals = pd.read_csv(filename)
            df = pd.DataFrame(consumer_list, columns=['Consumer'])
            df.to_csv(f'{output_dir}/consumer_list_iseed{config.iseed}.csv', index=False)

        pce = pauli_correlation_encode(m, n_qubits, k)

        if config.verbose:
            show_observable(pce)
        for alphasc in alphascs:
            alpha = alphasc * n_qubits ** np.floor(k / 2)

            for beta in betas:
                for iinit in range(ninit2):
                    theta0 = sample_init(
                        config.readmode,
                        rng,
                        ansatz.get_parameter_count(),
                        init_para
                    )
                    result, history, elapsed_time = read_optimize_fast(
                        theta0, config, dJ, dhex, n_qubits, k, ansatz,
                        pce, alphasc, beta, Cmin, Cmax, frob_norm, shift, iinit, output_dir, USE_BIAS
                    )

                    mineng, minnum = save_results_fast(
                        output_dir,
                        result,
                        history,
                        dJ, dhex,
                        ansatz,
                        pce,
                        n_qubits,
                        k,
                        Cmin, Cmax,
                        frob_norm, shift,
                        m,
                        alphasc, beta,
                        elapsed_time,
                        iinit,
                        config,
                        USE_BIAS,
                    )

                    print(alphasc, beta, iprob, iinit, mineng, minnum)
