import time, json, csv
from qulacs import ParametricQuantumCircuit, QuantumState, PauliOperator, Observable
from scipy.optimize import minimize
from power import power_loss_func_correct, sk_cost_func_fast
from src.core.utils import spin_to_number
from src.domain.loss.power_loss_func_fast import  power_loss_func_fast, power_loss_func_bias
import numpy as np

def read_optimize_fast(theta0, config, J, h, n_qubits, k, ansatz, hamiltonian,
                       alphasc, beta, Cmin, Cmax, frob_norm, shift,
                       iinit, output_dir):
    method = config.method
    verbose = config.verbose
    USE_BACKPROP = config.backprop
    USE_BIAS = config.bias
    maxiter = config.maxiter
    alpha = alphasc * n_qubits ** np.floor(k / 2)

    history = []
    best_cost = None
    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)

    # --- 勾配（Jacobian）関数 ---
    
    if USE_BIAS:
        def loss(params):
            theta = params[:-1]   # 最後以外がtheta
            bias = params[-1]     # 最後がbias
            loss, _ = power_loss_func_bias(J, h, n_qubits, theta, bias, ansatz, hamiltonian, alpha, beta)
            return loss

        bias0 = 0.0
        theta = np.concatenate([theta0, [bias0]])
        if USE_BACKPROP:
            def gradient(params):
                grad = backprop(params, n_qubits, len(h), ansatz=ansatz,
                                W=J, h=h, hamiltonian=hamiltonian, alpha=alpha, beta=beta, nu=1.0)
                return grad.flatten()
        else:
            gradient=None
        loss0, exp0 = power_loss_func_bias(J, h, n_qubits, theta0, bias0, ansatz, hamiltonian, alpha, beta)
        history = [(theta0.copy(), loss0, exp0)]
    else:
        def loss(params):
            loss, _ = power_loss_func_fast(J, h, n_qubits, params, ansatz, hamiltonian, alpha, beta)
            return loss

        theta = theta0
        if USE_BACKPROP:
            def gradient(params):
                grad = backprop(params, n_qubits, len(h), ansatz=ansatz,
                                W=J, h=h, hamiltonian=hamiltonian, alpha=alpha, beta=beta, nu=1.0)
                return grad.flatten()
        else:
            gradient=None
        loss0, exp0 = power_loss_func_fast(J, h, n_qubits, theta0, ansatz, hamiltonian, alpha, beta)
    history = [(theta0.copy(), loss0, exp0)]

    if verbose:
        spin_config = np.sign(exp0)
        cost0 = sk_cost_func_fast(J, h, spin_config)
        str_backprop = ''
        str_bias = ''
        if USE_BACKPROP: str_backprop = '_backprop'
        if USE_BIAS: str_backprop = '_bias'
        csv_path = f"{output_dir}/energy_progress{str_backprop}{str_bias}_alphasc{alphasc}_beta{beta}_init{iinit}.csv"
        with open(csv_path, mode='w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(['Iteration', 'Energy', 'Loss Function'])
            writer.writerow([0, norm(cost0), norm(loss0)])
        print(f"[Init] loss={norm(loss0):.6f}")

    def callback(xk):
        nonlocal best_cost
        if USE_BIAS:
            loss, exp_val = power_loss_func_bias(J, h, n_qubits, xk[:-1], xk[-1], ansatz, hamiltonian, alpha, beta)
        else:
            loss, exp_val = power_loss_func_fast(J, h, n_qubits, xk, ansatz, hamiltonian, alpha, beta)
        history.append((xk.copy(), loss, exp_val))
        if verbose:
            spin_config = np.sign(exp_val)
            cost = sk_cost_func_fast(J, h, spin_config)
            norm_cost = norm(cost)
            norm_loss = norm(loss)
            print(f"[Iter {len(history)-1}] loss={norm_loss:.6f}, cost={norm_cost:.6f}, number={spin_to_number(spin_config)}")

            if best_cost is None or norm_cost < best_cost:
                best_cost = norm_cost
                log_data = {
                    "iter": len(history)-1,
                    "loss": float(norm_loss),
                    "cost": float(norm_cost),
                    "parameters": xk.tolist(),
                    "number for cost": int(spin_to_number(spin_config)),
                }
                with open(f"{output_dir}/progress{str_backprop}{str_bias}_alphasc{alphasc}_beta{beta}_init{iinit}.json", "w") as f:
                    json.dump(log_data, f, indent=2)
            with open(csv_path, mode='a', newline='\n', encoding='utf-8') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([len(history)-1, norm_cost, norm_loss])

    start_time = time.time()
    result = minimize(loss, theta, jac=gradient, callback=callback, method=method,
                      options={"disp": verbose, "maxiter": maxiter})

    end_time = time.time()
    elapsed_time = end_time - start_time

    return result, history, elapsed_time

def backprop(parameters, n, m, ansatz, W, h, hamiltonian, alpha, beta, nu):
    layer = ansatz.depth
    p = len(parameters)
    dEi_dthetaj = np.zeros((m,p))

#    circuit = ParametricQuantumCircuit(n)
    ansatz.set_parameter(parameters)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    # Calculate the expectation values for each term in the Hamiltonian (from power.py)
    EV_array = np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

    # Tanh values
    tanh_vals = np.tanh(alpha * EV_array)

    # ---- 2. Partial derivative: dLoss/d(exp_value_i) ----
    dL_dEi = np.zeros_like(EV_array)

    # Interaction term derivative
    for i in range(m):
        for j in range(i):
            d_tanh_i = alpha * (1 - tanh_vals[i]**2)
            d_tanh_j = alpha * (1 - tanh_vals[j]**2)

            dL_dEi[i] += W[i, j] * tanh_vals[j] * d_tanh_i
            dL_dEi[j] += W[i, j] * tanh_vals[i] * d_tanh_j

    # Node weight term derivative
    dL_dEi += h * alpha * (1 - tanh_vals**2)

    # Regularizer term derivative
    for i in range(m):
        d_tanh = alpha * (1 - tanh_vals[i]**2)
        d_norm = (2 * tanh_vals[i] * d_tanh) / m
        dL_dEi[i] += beta * d_norm

    # --- Step 0: Pauli term list の作成（正しい方法） ---
    pauli_term_list = []
    for j in range(hamiltonian.get_term_count()):
        term = hamiltonian.get_term(j)
        pauli_str = term.get_pauli_string()   # 文字列形式に変換
        pauli_term_list.append(pauli_str)

    obs_all = Observable(n)
    for coef, term in zip(dL_dEi, pauli_term_list):
        if abs(float(coef)) > 1e-12:
            obs_all.add_operator(PauliOperator(term, float(coef)))
    # g[i] = dL/dθ_i
    g = ansatz.backprop(obs_all)
    
    # --- 結果を numpy 配列へ ---
    backprop_array = np.asarray(g, dtype=float)

    return backprop_array
