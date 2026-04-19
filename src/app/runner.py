# runner.py
import numpy as np
import pandas as pd
import os
import json

from src.core.graph_handler import prepare_int
from src.core.ansatz_factory import select_ansatz
from src.core.init_strategy import sample_init
from src.infra.input_handler import get_user_input, setup_output_dirs
from src.infra.result_handler import save_results_fast
from pce import pauli_correlation_encode, show_observable
from src.core.optimizer import read_optimize_fast

def run(config):
    rng = np.random.default_rng(config.iseed)

    # user input（ここだけ残す or 後でCLI化）
    it, nT, rate, n_qubits, k, m, depth, alphasc, beta, type_ansatz = get_user_input()

    if config.learn:
        alphascs = [0.01, 0.1, 0.5, 1.0, 1.5]
        betas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    else:
        alphascs, betas = [alphasc], [beta]

    output_dir, _ = setup_output_dirs(
        config.learn, "new" if config.use_new else "old",
        config.nprob, config.ninit, it, nT, rate, m,
        type_ansatz, n_qubits, k, depth,
        config.method, config.iseed, alphasc, beta
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
        alpha = alphasc * n_qubits ** np.floor(k / 2)

        read_file = os.path.join(
            output_dir,
            f"progress_alpha{alpha}_beta{beta}_init{iinit}_iseed{config.iseed}.json"
        )

        with open(read_file, "r") as f:
            data = json.load(f)

        init_para = np.array(data["parameters"])

        output_dir = os.path.join(output_dir, "read")
        os.makedirs(output_dir, exist_ok=True)

    for iprob in range(config.nprob):

        dJ, dhex, Cmin, Cmax, frob_norm, shift, consumer_list = prepare_int(
            "power_opt", m, it, nT, rate,
            rng, config.use_new, config.iseed
        )

        if not config.learn:
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
                        theta0,
                        config.method,
                        dJ, dhex,
                        n_qubits,
                        ansatz,
                        pce,
                        alpha, beta,
                        config.verbose,
                        Cmin, Cmax,
                        frob_norm, shift,
                        iinit,
                        output_dir,
                        config.backprop,
                        config.maxiter
                    )

                    mineng, minnum = save_results_fast(
                        output_dir,
                        result,
                        history,
                        dJ, dhex,
                        ansatz,
                        pce,
                        n_qubits,
                        Cmin, Cmax,
                        frob_norm, shift,
                        m,
                        alpha, beta,
                        elapsed_time,
                        iinit,
                        config.iseed,
                        config.verbose,
                        config.learn,
                        config.backprop
                    )

                    print(alphasc, beta, iprob, iinit, mineng, minnum)
