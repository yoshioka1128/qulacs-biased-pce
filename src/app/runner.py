# runner.py
import numpy as np
import pandas as pd
import os, json, itertools, shutil, json

from src.core.graph_handler import prepare_int
from src.core.ansatz_factory import select_ansatz
from src.core.init_strategy import sample_init
from src.infra.input_handler import setup_output_dirs
from src.infra.result_handler import save_results_fast
from pce import pauli_correlation_encode, show_observable
from src.core.optimizer import read_optimize_fast
from src.config.node_config import NODE_CONFIG
from src.analysis.loader import get_result_file_from_node_config
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_results

def run(config, args):
    node_cfg = NODE_CONFIG[args.m, args.rate, args.mode]

    alphasc = args.alphasc if args.alphasc is not None else node_cfg.alphasc
    beta = args.beta if args.beta is not None else node_cfg.beta

    params = {
        "m": args.m,
        "n_qubits": args.n_qubits or node_cfg.n_qubits,
        "k": args.k or node_cfg.k,
        "depth": args.depth or node_cfg.depth,
        "type_ansatz": args.type_ansatz,
        "itime": args.itime,
        "nT": args.nT,
        "rate": args.rate,
        "mode": args.mode,
#        "bias": args.bias,
        "alphasc": alphasc,
        "beta": beta,
    }

    if args.readmode:
        params["nT"] = 1

    if args.batch:
        run_batch(config, node_cfg, params)
    else:
        run_single(config, node_cfg, params)

def run_batch(config, node_cfg, params):
    beta_list = [-0.1, 0.0, 0.1, 0.2]
#    alpha_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    alpha_list = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for beta, alphasc in itertools.product(beta_list, alpha_list):
        print(f"\n=== beta={beta}, alphasc={alphasc} ===")

        batch_params = params.copy()
        batch_params["alphasc"] = alphasc
        batch_params["beta"] = beta

        run_single(config, node_cfg, batch_params)

def run_single(config, node_cfg, params):
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

    mode = params["mode"]

    alphasc = params["alphasc"]
    beta = params["beta"]
    if config.learn:
        alphascs = [0.01, 0.1, 0.5, 1.0, 1.5]
        betas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    else:
        alphascs, betas = [alphasc], [beta]

    output_dir = setup_output_dirs(
        config, "new", it, nT, rate, m,
        type_ansatz, n_qubits, k, depth,
    )

    ansatz = select_ansatz(type_ansatz, n_qubits, depth)

    if config.verbose:
        print(ansatz.get_qulacs_circuit())

    init_para = None
    ninit2 = config.ninit

    # readmode
    if config.readmode:
        ninit2 = 1
        iinit = node_cfg.iinit

        suffix = []
        if config.backprop: suffix.append("backprop")
        if mode != "nobias": suffix.append(mode)
        suffix = "_" + "_".join(suffix) if suffix else ""

#        read_dir = (f'outputs/power_opt/time1_nT24_rate{rate}_{m}nodes_{n_qubits}qubits_{k}body_'
#                      f'ninit5_depth5_{type_ansatz}_method{config.method}_iseed{config.iseed}/')
#        read_file = os.path.join(
#            read_dir,
#            f"results{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.json"
#        )
        read_dir, read_file = get_result_file_from_node_config(
            m, rate, mode, node_cfg.method, node_cfg.iseed, node_cfg.type_ansatz,
        )
        dst_file = os.path.join(output_dir, os.path.basename(read_file))
        print('copy from', read_file, 'to', dst_file)
        shutil.copy2(read_file, dst_file)
        with open(read_file, "r") as f:
            data = json.load(f)

        init_para = np.array(data["Minimum Parameters"])

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
                    init_para = sample_init(config.readmode, mode, rng, ansatz.get_parameter_count(), init_para)
                    result, history, elapsed_time = read_optimize_fast(
                        init_para, config, dJ, dhex, n_qubits, k, ansatz,
                        pce, alphasc, beta, Cmin, Cmax, frob_norm, shift, iinit, output_dir,
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
                    )

                    print(alphasc, beta, iprob, iinit, mineng, minnum)
