import time, json, csv
from qulacs import ParametricQuantumCircuit, QuantumState, PauliOperator, Observable
from scipy.optimize import minimize
from src.core.utils import spin_to_number
from src.core.expectation import compute_expectation
from src.domain.loss.power_loss import  compute_loss, compute_loss_bias_x, compute_loss_bias_xy
from src.domain.loss.power_grad_loss import  backprop, backprop_bias_x, backprop_bias_y
from src.domain.cost.power_cost import  compute_cost
import numpy as np

# =========================
# Optimization wrapper
# =========================
def read_optimize_fast(
        init_para, config, J, h, n_qubits, k, ansatz, hamiltonian,
        alphasc, beta, Cmin, Cmax, frob_norm, shift,
        iinit, output_dir,
):
    # ---- config ----
    mode = config.mode
    method = config.method
    verbose = config.verbose
    USE_BACKPROP = config.backprop
    maxiter = config.maxiter
    USE_BIAS = (mode != "nobias")
#    reg_type = None if mode == "nobias" else mode[-1]
#    reg = "" if reg_type is None else f"_reg_type{reg_type}"
#    reg_type = config.reg_type
#    if USE_BIAS: reg = f'_reg_type{reg_type}'
#    else: reg = ''

    alpha = alphasc * n_qubits ** np.floor(k / 2)

    norm = lambda x: (x * frob_norm + shift - Cmin) / (Cmax - Cmin)

    # =========================
    # Parameter handling
    # =========================
    def split_params(params):
        if USE_BIAS: return params[:-1], params[-1]
        return params, None

#    def merge_params(theta, bias):
#        if USE_BIAS: return np.concatenate([theta, [bias]])
#        return theta
#    init_para = merge_params(theta0, 0.0)

    # =========================
    # Loss / Gradient
    # =========================
    if mode == "bias_x":
#        if reg_type == 'x':
        def loss_fn(params):
            theta, bias = split_params(params)
            loss, _ = compute_loss_bias_x(
                J, h, n_qubits, theta, ansatz, hamiltonian,
                alpha, beta,
                bias=bias
            )
            return loss
        def grad_fn_bias(params):
            grad = backprop_bias_x(
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
    elif mode == "bias_y":
#        elif reg_type == 'y':
        def loss_fn(params):
            theta, bias = split_params(params)
            loss, _ = compute_loss_bias_xy(
                J, h, n_qubits, theta, ansatz, hamiltonian,
                alpha, beta,
                bias=bias,
            )
            return loss
        def grad_fn_bias(params):
            grad = backprop_bias_y(
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
    else: # USE_BIAS = False
        def loss_fn(params):
            theta, bias = split_params(params)
            loss, _ = compute_loss(
                J, h, n_qubits, theta, ansatz, hamiltonian,
                alpha, beta,
            )
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

    # =========================
    # Logging setup
    # =========================
    history = []
    best_cost = None

    suffix = []
    if USE_BACKPROP: suffix.append("backprop")
    if mode != "nobias": suffix.append(mode)
    suffix = "_" + "_".join(suffix) if suffix else ""

    csv_path = f"{output_dir}/energy_progress{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.csv"
    json_path = f"{output_dir}/progress{suffix}_alphasc{alphasc}_beta{beta}_init{iinit}.json"

    # ---- initial evaluation ----
    theta, bias = split_params(init_para)
    loss0 = loss_fn(init_para)
    exp0 = compute_expectation(n_qubits, theta, ansatz, hamiltonian)

    if USE_BIAS: history.append((init_para.copy(), loss0, exp0, bias)) # nparam + 1
    else: history.append((theta.copy(), loss0, exp0)) # nparam

    if verbose:
        z = alpha * exp0 + (bias if bias is not None else 0.0)
        spin_config = np.sign(z)
        cost0 = compute_cost(J, h, spin_config)

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
        loss = loss_fn(params)
        exp_val = compute_expectation(n_qubits, theta, ansatz, hamiltonian)

        if USE_BIAS:
            history.append((params.copy(), loss, exp_val, bias))
        else:
            history.append((params.copy(), loss, exp_val))

        if not verbose:
            return
        z = alpha * exp_val + (bias if bias is not None else 0.0)
        spin_config = np.sign(z)
        cost = compute_cost(J, h, spin_config)

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
                init_para,
                jac=grad_fn_bias,
                callback=callback,
                method=method,
                options={"disp": verbose, "maxiter": maxiter},
            )
        else:
            result = minimize(
                loss_fn,
                init_para,
                jac=grad_fn,
                callback=callback,
                method=method,
                options={"disp": verbose, "maxiter": maxiter},
            )
    else:
        result = minimize(
            loss_fn,
            init_para,
            callback=callback,
            method=method,
            options={"disp": verbose, "maxiter": maxiter},
        )

    elapsed = time.time() - start

    return result, history, elapsed

    return g_theta


def greedy_ising(J, h, z0=None):
    """
    Apply a greedy spin-flip algorithm to the Ising model (J, h).
    Spins are treated directly as z ∈ {-1, +1}.

    Parameters
    ----------
    J : ndarray (N, N)
        Symmetric interaction matrix (diagonal elements assumed to be 0)
    h : ndarray (N,)
        Local magnetic field
    z0 : ndarray (N,), optional
        Initial spin configuration (elements can be ±1 or 0).
        If 0, the value is first greedily assigned to -1 or +1.
        If None, all spins are initialized to +1.

    Returns
    -------
    z : ndarray (N,)
        Final spin configuration (±1)
    E : float
        Final energy
    """
    N = len(h)

    # Initialize spins
    if z0 is None:
        z = np.ones(N, dtype=int)
    else:
        z = np.array(z0, dtype=int).copy()
        if not np.all(np.isin(z, [-1, 0, 1])):
            raise ValueError("Elements of z0 must be -1, 0, or +1")

    # --- Step 1: Assign spins where z=0 greedily ---
    for i in range(N):
        if z[i] == 0:
            # Compute effective local field using currently assigned spins
            effective_field = h[i] + np.dot(J[i], z * (z != 0))  # ignore unassigned spins
            if effective_field > 0:
                z[i] = -1
            elif effective_field < 0:
                z[i] = +1
            else:
                z[i] = np.random.choice([-1, 1])  # tie-break

    # --- Step 2: Apply standard greedy spin-flip ---
    chosen = np.zeros(N, dtype=bool)
    local_field = h + J @ z

    while True:
        delta = np.full(N, np.inf)
        for j in range(N):
            if not chosen[j]:
                # ΔE_j = -2 z_j (h_j + sum_i J_ij z_i)
                delta[j] = -2 * z[j] * local_field[j]

        j_best = np.argmin(delta)
        if delta[j_best] >= 0:
            break

        # スピン反転
        z[j_best] *= -1
        chosen[j_best] = True

        # local_field を更新
        local_field += 2 * J[:, j_best] * z[j_best]

    # エネルギー計算
    E = float(z @ np.tril(J, -1) @ z + h @ z)

    return z.tolist(), E
