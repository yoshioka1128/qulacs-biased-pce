import time, json, csv
from qulacs import ParametricQuantumCircuit, QuantumState, PauliOperator, Observable
from scipy.optimize import minimize
from power import power_loss_func, power_loss_func_correct, power_loss_func_fast, sk_cost_func_fast
from src.core.utils import spin_to_number
import numpy as np

def read_optimize_fast(theta0, method, J, h, n_qubits, ansatz, hamiltonian,
                       alpha, beta, verbose, Cmin, Cmax, frob_norm, shift,
                       iinit, output_dir, USE_BACKPROP='n', maxiter=10000):
    history = []
    best_cost = None
    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)

    # --- Add the initial point to history so that it's never empty ---
    loss0, exp0 = power_loss_func_fast(J, h, n_qubits, theta0, ansatz, hamiltonian, alpha, beta)
    history = [(theta0.copy(), loss0, exp0)]
    # -----------------------------------------------------------------
    if verbose:
        spin_config = np.sign(exp0)
        cost0 = sk_cost_func_fast(J, h, spin_config)
        str_backprop = ''
        if USE_BACKPROP: str_backprop = '_backprop'
        csv_path = f"{output_dir}/energy{str_backprop}_progress_alpha{alpha}_beta{beta}_init{iinit}.csv"
        with open(csv_path, mode='w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(['Iteration', 'Energy', 'Loss Function'])
            writer.writerow([0, norm(cost0), norm(loss0)])
        print(f"[Init] loss={norm(loss0):.6f}")

    def cost(params):
        loss, _ = power_loss_func_fast(J, h, n_qubits, params, ansatz, hamiltonian, alpha, beta)
        return loss

    # --- 勾配（Jacobian）関数 ---
    def gradient(params):
        """
        backprop() を用いて ∂loss/∂params を計算
        """
        # MaxCutの場合、hamiltonian は重み行列 W に対応
        # alpha, beta, nu のようなパラメータは上位から渡す
        # 以下は backprop の呼び出しに合わせて調整してください
        grad = backprop(params, n_qubits, len(h), ansatz=ansatz,
                        W=J, h=h, hamiltonian=hamiltonian, alpha=alpha, beta=beta, nu=1.0)
        return grad.flatten()

    def callback(xk):
        nonlocal best_cost
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
                with open(f"{output_dir}/progress{str_backprop}_alpha{alpha}_beta{beta}_init{iinit}.json", "w") as f:
                    json.dump(log_data, f, indent=2)
            with open(csv_path, mode='a', newline='\n', encoding='utf-8') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([len(history)-1, norm_cost, norm_loss])

    start_time = time.time()
    # bounds = [(-2.0*np.pi, 4.0*np.pi) for i in range(len(theta0))]
    if USE_BACKPROP:
        result = minimize(cost, theta0, jac=gradient, callback=callback, method=method,
                          options={"disp": verbose, "maxiter": maxiter})
    else:
        result = minimize(cost, theta0, callback=callback, method=method, #bounds = bounds, # BFGS or SLSQP
                          options={"disp": verbose, "maxiter": maxiter}) # default gtol = 1e-5 in BFGS

    end_time = time.time()
    elapsed_time = end_time - start_time

    return result, history, elapsed_time


def Each_layer(circuit, parameters, i, n):

    # This is to implement the single-qubit rotation gates.
    for j in range(n):
        circuit.add_parametric_RX_gate(j,parameters[i*int(n*(n+3)/2)+j])
    for j in range(n):
        circuit.add_parametric_RZ_gate(j,parameters[i*int(n*(n+3)/2)+n+j])

    # This is to implement all-to-all R_zz gates.
    temp = 0
    for j in range(n):
        for k in range(j):
            circuit.add_CNOT_gate(j,k)
            circuit.add_parametric_RZ_gate(k,parameters[i*int(n*(n+3)/2)+2*n+temp])
            circuit.add_CNOT_gate(j,k)
            temp = temp + 1

    return circuit

# The following function is to calculate the gradient using back propagation method.
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

#    for j in range(m):
#        grad = ansatz.backprop(hamiltonian.get_term(j))
#        for i in range(p):
#            dEi_dthetaj[j,i] = grad[i]
#    backprop_array = np.dot(dL_dEi, dEi_dthetaj)

    
    return backprop_array
