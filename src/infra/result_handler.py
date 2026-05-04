import json
import csv
import numpy as np
from src.core.utils import get_binary_solution, spin_to_number, convert_seconds_to_hms
from src.domain.cost.power_cost import compute_cost

def save_results_fast(output_dir, result, history, dJ, dhex, ansatz, hamiltonian, n_qubits, k, Cmin, Cmax, frob_norm, shift, n_nodes,
                      alphasc, beta, elapsed_time, iinit, config):

    verbose = config.verbose
    LEARN = config.learn
    USE_BACKPROP = config.backprop
    bias_mode = config.bias_mode
    USE_BIAS = (bias_mode != "nobias")

    loss_history = [h[1] for h in history]
    exp_history  = [h[2] for h in history]
    alpha = alphasc * n_qubits ** np.floor(k / 2)
    bit_history  = [
        get_binary_solution(n_nodes, h[0], ansatz, hamiltonian, n_qubits, bias_mode, alpha)
        for h in history
    ]
    if USE_BIAS: bias_history = [h[3] for h in history]
    
    cost_history = [compute_cost(dJ, dhex, x) for x in bit_history]
    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)
    min_idx = int(np.argmin(cost_history))
    min_params = history[min_idx][0]
    min_spin = bit_history[min_idx]
    hours, minutes, seconds = convert_seconds_to_hms(elapsed_time)

    mineng = norm(cost_history[min_idx])
    minnum = spin_to_number(min_spin)

    result_dict = {
        "Calculated Minimum Energy [norm, row]": [mineng, cost_history[min_idx]],
        "Corresponding loss function": norm(loss_history[min_idx]),
        "Solution for Minimum Energy": min_spin.tolist(),
        "Corresponding exp value": exp_history[min_idx].tolist(),
        "Number for Minimum Energy": minnum,
        "Elapsed Time [seconds]": elapsed_time,
        "Elapsed Time [hours, minutes, seconds]": [hours, minutes, round(seconds, 2)],
        "Current function value": result.fun,
        "Iterations": result.nit,
        "Function Evaluations, Estimation": [result.nfev, result.njev*(len(min_params.tolist())+1)],
        "Gradient Evaluations": result.njev,
        "Number of Parameters": len(min_params.tolist()),
        "Minimum Parameters": min_params.tolist(),
        "Cmin, Cmax, frob_norm, shift": [Cmin, Cmax, frob_norm, shift],
    }

    if verbose:
        keys_to_show = ["Corresponding loss function",
                        "Solution for Minimum Energy", "Corresponding exp value",
                        "Iterations", "Number for Minimum Energy", "Function Evaluations, Estimation",
                        "Gradient Evaluations", "Number of Parameters"]
        for k in keys_to_show:
            print(f"{k}:",result_dict[k])

    # 出力
    if LEARN == 0:
        suffix = []
        if USE_BACKPROP: suffix.append("backprop")
        if bias_mode != "nobias": suffix.append(bias_mode)
        suffix = "_" + "_".join(suffix) if suffix else ""

#        str_backprop = ''
#        str_bias = ''
#        if USE_BACKPROP: str_backprop = '_backprop'
#        if USE_BIAS: str_bias = '_bias'
        with open(f"{output_dir}/results{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.json", 'w') as f:
            json.dump(result_dict, f, indent=4)

        with open(f"{output_dir}/energy{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.csv", mode='w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")

            if USE_BIAS:
                writer.writerow(['Iteration', 'Energy', 'Loss Function', 'Bias'])
                for i, (c, l, b) in enumerate(zip(cost_history, loss_history, bias_history)):
                    writer.writerow([i, norm(c), norm(l), b])
            else:
                writer.writerow(['Iteration', 'Energy', 'Loss Function'])
                for i, (c, l) in enumerate(zip(cost_history, loss_history)):
                    writer.writerow([i, norm(c), norm(l)])

    return mineng, minnum
