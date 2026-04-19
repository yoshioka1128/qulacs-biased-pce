import json
import csv
import numpy as np
from power import sk_cost_func, sk_cost_func_fast
from src.core.utils import get_binary_solution, get_binary_solution_fast, spin_to_number, convert_seconds_to_hms

def save_results_fast(output_dir, result, history, dJ, dhex, ansatz, hamiltonian, n_qubits, Cmin, Cmax, frob_norm, shift, n_nodes,
                 alphasc, beta, elapsed_time, iinit, verbose, LEARN, USE_BACKPROP='n'):
    loss_history = [loss for _, loss, _ in history]
    bit_history = [get_binary_solution_fast(n_nodes, p, ansatz, hamiltonian, n_qubits) for p, _, _ in history]
    cost_history = [sk_cost_func_fast(dJ, dhex, x) for x in bit_history]
    exp_history = [exp for _, _, exp in history]
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
        str_backprop=''
        if USE_BACKPROP: str_backprop='_backprop'
        with open(f"{output_dir}/results{str_backprop}_alphasc{alphasc}_beta{beta}_init{iinit}.json", 'w') as f:
            json.dump(result_dict, f, indent=4)

        with open(f'{output_dir}/energy{str_backprop}_alphasc{alphasc}_beta{beta}_init{iinit}.csv', mode='w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(['Iteration', 'Energy', 'Loss Function'])
            for i, (c, l) in enumerate(zip(cost_history, loss_history)):
                writer.writerow([i, norm(c), norm(l)])

    return mineng, minnum
