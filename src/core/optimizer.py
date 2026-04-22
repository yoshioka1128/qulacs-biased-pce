import time, json, csv
from qulacs import ParametricQuantumCircuit, QuantumState, PauliOperator, Observable
from scipy.optimize import minimize
from power import power_loss_func_correct, sk_cost_func_fast
from src.core.utils import spin_to_number
from src.domain.loss.power_loss_func_fast import  power_loss_func_fast, power_loss_func_bias, compute_loss
import numpy as np

# =========================
# Optimization wrapper
# =========================
def read_optimize_fast(
        theta0, config, J, h, n_qubits, k, ansatz, hamiltonian,
        alphasc, beta, Cmin, Cmax, frob_norm, shift,
        iinit, output_dir, USE_BIAS
):
    # ---- config ----
    method = config.method
    verbose = config.verbose
    USE_BACKPROP = config.backprop
#    USE_BIAS = config.bias
    maxiter = config.maxiter

    alpha = alphasc * n_qubits ** np.floor(k / 2)

    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)

    # =========================
    # Parameter handling
    # =========================
    def split_params(params):
        if USE_BIAS: return params[:-1], params[-1]
        return params, None

    def merge_params(theta, bias):
        if USE_BIAS: return np.concatenate([theta, [bias]])
        return theta

    theta_init = merge_params(theta0, 0.0)

    # =========================
    # Loss / Gradient
    # =========================
    def loss_fn(params):
        theta, bias = split_params(params)
        loss, _ = compute_loss(J, h, n_qubits, theta, ansatz, hamiltonian, alpha, beta, bias)
        return loss

    def grad_fn(params):
        grad = backprop(
            params,
            n_qubits,
            len(h),
            ansatz=ansatz,
            W=J,
            h=h,
            hamiltonian=hamiltonian,
            alpha=alpha,
            beta=beta,
            nu=1.0,
        )
        return grad.flatten()

    def grad_fn_bias(params):
        grad = backprop_bias(
            params,
            n_qubits,
            len(h),
            ansatz=ansatz,
            W=J,
            h=h,
            hamiltonian=hamiltonian,
            alpha=alpha,
            beta=beta,
            nu=1.0,
        )
        return grad.flatten()

    # =========================
    # Logging setup
    # =========================
    history = []
    best_cost = None

    suffix = []
    if USE_BACKPROP: suffix.append("backprop")
    if USE_BIAS: suffix.append("bias")
    suffix = "_" + "_".join(suffix) if suffix else ""

    csv_path = f"{output_dir}/energy_progress{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.csv"
    json_path = f"{output_dir}/progress{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.json"

    # ---- initial evaluation ----
    theta, bias = split_params(theta_init)
    loss0, exp0 = compute_loss(J, h, n_qubits, theta, ansatz, hamiltonian, alpha, beta, bias)

    if USE_BIAS: history.append((theta_init.copy(), loss0, exp0, bias)) # nparam + 1
    else: history.append((theta.copy(), loss0, exp0)) # nparam

    if verbose:
        z = alpha * exp0 + (bias if bias is not None else 0.0)
        spin_config = np.sign(z)
        cost0 = sk_cost_func_fast(J, h, spin_config)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")

            if USE_BIAS:
                writer.writerow(["Iteration", "Energy", "Loss Function", "Bias"])
                writer.writerow([0, norm(cost0), norm(loss0), bias])
            else:
                writer.writerow(["Iteration", "Energy", "Loss Function"])
                writer.writerow([0, norm(cost0), norm(loss0)])

        print(f"[Init] loss={norm(loss0):.6f}")

    # =========================
    # Callback
    # =========================
    def callback(params):
        nonlocal best_cost

        theta, bias = split_params(params)
        loss, exp_val = compute_loss(J, h, n_qubits, theta, ansatz, hamiltonian, alpha, beta, bias)

        if USE_BIAS:
            history.append((params.copy(), loss, exp_val, bias))
        else:
            history.append((params.copy(), loss, exp_val))

        if not verbose:
            return
        z = alpha * exp_val + (bias if bias is not None else 0.0)
        spin_config = np.sign(z)
        cost = sk_cost_func_fast(J, h, spin_config)

        norm_cost = norm(cost)
        norm_loss = norm(loss)

        print(
            f"[Iter {len(history)-1}] "
            f"loss={norm_loss:.6f}, cost={norm_cost:.6f}, bias={bias} "
            f"number={spin_to_number(spin_config)}"
        )

        # ---- best update ----
        if best_cost is None or norm_cost < best_cost:
            best_cost = norm_cost
            log_data = {
                "iter": len(history) - 1,
                "loss": float(norm_loss),
                "cost": float(norm_cost),
                "parameters": params.tolist(),
                "number for cost": int(spin_to_number(spin_config)),
            }
            with open(json_path, "w") as f:
                json.dump(log_data, f, indent=2)

        # ---- append csv ----
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")

            if USE_BIAS:
                writer.writerow([len(history)-1, norm_cost, norm_loss, bias])
            else:
                writer.writerow([len(history)-1, norm_cost, norm_loss])

    # =========================
    # Run optimization
    # =========================
    start = time.time()
    if USE_BACKPROP:
        if USE_BIAS:
            result = minimize(
                loss_fn,
                theta_init,
                jac=grad_fn_bias,
                callback=callback,
                method=method,
                options={"disp": verbose, "maxiter": maxiter},
            )
        else:
            result = minimize(
                loss_fn,
                theta_init,
                jac=grad_fn,
                callback=callback,
                method=method,
                options={"disp": verbose, "maxiter": maxiter},
            )
    else:
        result = minimize(
            loss_fn,
            theta_init,
            callback=callback,
            method=method,
            options={"disp": verbose, "maxiter": maxiter},
        )

    elapsed = time.time() - start

    return result, history, elapsed


def backprop_bias_old(parameters, n, m, ansatz, W, h, hamiltonian, alpha, beta, nu, use_bias=True):
    layer = ansatz.depth

    if use_bias:
        theta = parameters[:-1]
        bias = parameters[-1]
    else:
        theta = parameters
        bias = 0.0

    p = len(theta)

    # --- Forward ---
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    EV_array = np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

    z = alpha * EV_array + bias
    tanh_vals = np.tanh(z)

    # =========================
    # dL / dE_i
    # =========================
    dL_dEi = np.zeros_like(EV_array)

    for i in range(m):
        for j in range(i):
            d_tanh_i = alpha * (1 - tanh_vals[i]**2)
            d_tanh_j = alpha * (1 - tanh_vals[j]**2)

            dL_dEi[i] += W[i, j] * tanh_vals[j] * d_tanh_i
            dL_dEi[j] += W[i, j] * tanh_vals[i] * d_tanh_j

    dL_dEi += h * alpha * (1 - tanh_vals**2)

    for i in range(m):
        d_tanh = alpha * (1 - tanh_vals[i]**2)
        d_norm = (2 * tanh_vals[i] * d_tanh) / m
        dL_dEi[i] += beta * d_norm

    # =========================
    # Observable構築
    # =========================
    pauli_term_list = [
        hamiltonian.get_term(j).get_pauli_string()
        for j in range(hamiltonian.get_term_count())
    ]

    obs_all = Observable(n)
    for coef, term in zip(dL_dEi, pauli_term_list):
        if abs(float(coef)) > 1e-12:
            obs_all.add_operator(PauliOperator(term, float(coef)))

    # =========================
    # θ 勾配
    # =========================
    g_theta = np.asarray(ansatz.backprop(obs_all), dtype=float)

    # =========================
    # bias 勾配（追加）
    # =========================
    if use_bias:
        dL_dbias = 0.0

        # Interaction term
        for i in range(m):
            for j in range(i):
                d_tanh_i = (1 - tanh_vals[i]**2)
                d_tanh_j = (1 - tanh_vals[j]**2)

                dL_dbias += W[i, j] * (
                    tanh_vals[j] * d_tanh_i +
                    tanh_vals[i] * d_tanh_j
                )

        # Node term
        dL_dbias += np.sum(h * (1 - tanh_vals**2))

        # Regularizer
        for i in range(m):
            d_tanh = (1 - tanh_vals[i]**2)
            d_norm = (2 * tanh_vals[i] * d_tanh) / m
            dL_dbias += beta * d_norm

        # 結合
        return np.concatenate([g_theta, [dL_dbias]])

    return g_theta

def backprop_bias(parameters, n, m, ansatz, W, h, hamiltonian, alpha, beta, nu):
    theta = parameters[:-1]
    b = parameters[-1]
    
#    if use_bias:
#        theta = parameters[:-1]
#        b = parameters[-1]
#    else:
#        theta = parameters
#        b = 0.0

    # --- state ---
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    # --- expectation ---
    EV_array = np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

    # --- nonlinear ---
    y_vals = np.tanh(alpha * EV_array)          # 正則化用
    x_vals = np.tanh(alpha * EV_array + b)      # エネルギー用

    # --- derivatives ---
    dx_dE = alpha * (1 - x_vals**2)
    dy_dE = alpha * (1 - y_vals**2)

    # --- dL/dEi ---
    dL_dEi = np.zeros_like(EV_array)

    # --- interaction ---
    for i in range(m):
        for j in range(i):
            dL_dEi[i] += W[i, j] * x_vals[j] * dx_dE[i]
            dL_dEi[j] += W[i, j] * x_vals[i] * dx_dE[j]

    # --- linear term ---
    dL_dEi += h * dx_dE

    # --- regularization（ここが重要変更） ---
    dL_dEi += (2 * beta / m) * y_vals * dy_dE

    # --- Observable ---
    obs_all = Observable(n)
    for i in range(len(dL_dEi)):
        coef = float(dL_dEi[i])
        if abs(coef) > 1e-12:
            term = hamiltonian.get_term(i).get_pauli_string()
            obs_all.add_operator(PauliOperator(term, coef))

    # --- θ gradient ---
    g_theta = ansatz.backprop(obs_all)
    g_theta = np.asarray(g_theta, dtype=float)

    # --- bias gradient ---
    dL_db = np.sum(
        (np.dot(W, x_vals) + h) * (1 - x_vals**2)
    )
    return np.concatenate([g_theta, [dL_db]])


def backprop(parameters, n, m, ansatz, W, h, hamiltonian, alpha, beta, nu):
    theta = parameters

    # --- Forward ---
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    EV_array = np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

    # --- 明示的に分ける（重要） ---
    y_vals = np.tanh(alpha * EV_array)
    x_vals = y_vals  # bias=0 の場合

    dx_dE = alpha * (1 - x_vals**2)
    dy_dE = alpha * (1 - y_vals**2)

    # =========================
    # dL / dE_i
    # =========================
    dL_dEi = np.zeros_like(EV_array)

    # --- interaction ---
    for i in range(m):
        for j in range(i):
            dL_dEi[i] += W[i, j] * x_vals[j] * dx_dE[i]
            dL_dEi[j] += W[i, j] * x_vals[i] * dx_dE[j]

    # --- linear ---
    dL_dEi += h * dx_dE

    # --- regularization（yを使う）---
    dL_dEi += (2 * beta / m) * y_vals * dy_dE

    # =========================
    # Observable構築
    # =========================
    obs_all = Observable(n)
    for i, coef in enumerate(dL_dEi):
        if abs(float(coef)) > 1e-12:
            term = hamiltonian.get_term(i).get_pauli_string()
            obs_all.add_operator(PauliOperator(term, float(coef)))

    # =========================
    # θ 勾配
    # =========================
    g_theta = np.asarray(ansatz.backprop(obs_all), dtype=float)

    return g_theta


#def read_optimize_fast(theta0, config, J, h, n_qubits, k, ansatz, hamiltonian,
#                       alphasc, beta, Cmin, Cmax, frob_norm, shift,
#                       iinit, output_dir):
#    method = config.method
#    verbose = config.verbose
#    USE_BACKPROP = config.backprop
#    USE_BIAS = config.bias
#    maxiter = config.maxiter
#    alpha = alphasc * n_qubits ** np.floor(k / 2)
#
#    history = []
#    best_cost = None
#    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)
#
#    # --- 勾配（Jacobian）関数 ---
#    
#    if USE_BIAS:
#        def loss(params):
#            theta = params[:-1]   # 最後以外がtheta
#            bias = params[-1]     # 最後がbias
#            loss, _ = power_loss_func_bias(J, h, n_qubits, theta, bias, ansatz, hamiltonian, alpha, beta)
#            return loss
#
#        bias0 = 0.0
#        theta = np.concatenate([theta0, [bias0]])
#        if USE_BACKPROP:
#            def gradient(params):
#                grad = backprop(params, n_qubits, len(h), ansatz=ansatz,
#                                W=J, h=h, hamiltonian=hamiltonian, alpha=alpha, beta=beta, nu=1.0)
#                return grad.flatten()
#        else:
#            gradient=None
#        loss0, exp0 = power_loss_func_bias(J, h, n_qubits, theta0, bias0, ansatz, hamiltonian, alpha, beta)
#        history = [(theta0.copy(), loss0, exp0)]
#    else:
#        def loss(params):
#            loss, _ = power_loss_func_fast(J, h, n_qubits, params, ansatz, hamiltonian, alpha, beta)
#            return loss
#
#        theta = theta0
#        if USE_BACKPROP:
#            def gradient(params):
#                grad = backprop(params, n_qubits, len(h), ansatz=ansatz,
#                                W=J, h=h, hamiltonian=hamiltonian, alpha=alpha, beta=beta, nu=1.0)
#                return grad.flatten()
#        else:
#            gradient=None
#        loss0, exp0 = power_loss_func_fast(J, h, n_qubits, theta0, ansatz, hamiltonian, alpha, beta)
#    history = [(theta0.copy(), loss0, exp0)]
#
#    if verbose:
#        spin_config = np.sign(exp0)
#        cost0 = sk_cost_func_fast(J, h, spin_config)
#        str_backprop = ''
#        str_bias = ''
#        if USE_BACKPROP: str_backprop = '_backprop'
#        if USE_BIAS: str_backprop = '_bias'
#        csv_path = f"{output_dir}/energy_progress{str_backprop}{str_bias}_alphasc{alphasc}_beta{beta}_init{iinit}.csv"
#        with open(csv_path, mode='w', newline='\n', encoding='utf-8') as f:
#            writer = csv.writer(f, lineterminator="\n")
#            writer.writerow(['Iteration', 'Energy', 'Loss Function'])
#            writer.writerow([0, norm(cost0), norm(loss0)])
#        print(f"[Init] loss={norm(loss0):.6f}")
#
#    def callback(xk):
#        nonlocal best_cost
#        if USE_BIAS:
#            loss, exp_val = power_loss_func_bias(J, h, n_qubits, xk[:-1], xk[-1], ansatz, hamiltonian, alpha, beta)
#        else:
#            loss, exp_val = power_loss_func_fast(J, h, n_qubits, xk, ansatz, hamiltonian, alpha, beta)
#        history.append((xk.copy(), loss, exp_val))
#        if verbose:
#            spin_config = np.sign(exp_val)
#            cost = sk_cost_func_fast(J, h, spin_config)
#            norm_cost = norm(cost)
#            norm_loss = norm(loss)
#            print(f"[Iter {len(history)-1}] loss={norm_loss:.6f}, cost={norm_cost:.6f}, number={spin_to_number(spin_config)}")
#
#            if best_cost is None or norm_cost < best_cost:
#                best_cost = norm_cost
#                log_data = {
#                    "iter": len(history)-1,
#                    "loss": float(norm_loss),
#                    "cost": float(norm_cost),
#                    "parameters": xk.tolist(),
#                    "number for cost": int(spin_to_number(spin_config)),
#                }
#                with open(f"{output_dir}/progress{str_backprop}{str_bias}_alphasc{alphasc}_beta{beta}_init{iinit}.json", "w") as f:
#                    json.dump(log_data, f, indent=2)
#            with open(csv_path, mode='a', newline='\n', encoding='utf-8') as f:
#                writer = csv.writer(f, lineterminator="\n")
#                writer.writerow([len(history)-1, norm_cost, norm_loss])
#
#    start_time = time.time()
#    result = minimize(loss, theta, jac=gradient, callback=callback, method=method,
#                      options={"disp": verbose, "maxiter": maxiter})
#
#    end_time = time.time()
#    elapsed_time = end_time - start_time
#
#    return result, history, elapsed_time

#def backprop(parameters, n, m, ansatz, W, h, hamiltonian, alpha, beta, nu):
#    layer = ansatz.depth
#    p = len(parameters)
#    dEi_dthetaj = np.zeros((m,p))
#
##    circuit = ParametricQuantumCircuit(n)
#    ansatz.set_parameter(parameters)
#    state = QuantumState(n)
#    ansatz.update_quantum_state(state)
#
#    # Calculate the expectation values for each term in the Hamiltonian (from power.py)
#    EV_array = np.array([
#        hamiltonian.get_term(i).get_expectation_value(state).real
#        for i in range(hamiltonian.get_term_count())
#    ])
#
#    # Tanh values
#    tanh_vals = np.tanh(alpha * EV_array)
#
#    # ---- 2. Partial derivative: dLoss/d(exp_value_i) ----
#    dL_dEi = np.zeros_like(EV_array)
#
#    # Interaction term derivative
#    for i in range(m):
#        for j in range(i):
#            d_tanh_i = alpha * (1 - tanh_vals[i]**2)
#            d_tanh_j = alpha * (1 - tanh_vals[j]**2)
#
#            dL_dEi[i] += W[i, j] * tanh_vals[j] * d_tanh_i
#            dL_dEi[j] += W[i, j] * tanh_vals[i] * d_tanh_j
#
#    # Node weight term derivative
#    dL_dEi += h * alpha * (1 - tanh_vals**2)
#
#    # Regularizer term derivative
#    for i in range(m):
#        d_tanh = alpha * (1 - tanh_vals[i]**2)
#        d_norm = (2 * tanh_vals[i] * d_tanh) / m
#        dL_dEi[i] += beta * d_norm
#
#    # --- Step 0: Pauli term list の作成（正しい方法） ---
#    pauli_term_list = []
#    for j in range(hamiltonian.get_term_count()):
#        term = hamiltonian.get_term(j)
#        pauli_str = term.get_pauli_string()   # 文字列形式に変換
#        pauli_term_list.append(pauli_str)
#
#    obs_all = Observable(n)
#    for coef, term in zip(dL_dEi, pauli_term_list):
#        if abs(float(coef)) > 1e-12:
#            obs_all.add_operator(PauliOperator(term, float(coef)))
#    # g[i] = dL/dθ_i
#    g = ansatz.backprop(obs_all)
#    
#    # --- 結果を numpy 配列へ ---
#    backprop_array = np.asarray(g, dtype=float)
#
#    return backprop_array
