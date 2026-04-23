# power_grad_loss.py
import numpy as np

from qulacs import QuantumState
from qulacs import Observable
from qulacs import PauliOperator

def backprop_bias_x(parameters, n, m, ansatz, W, h, hamiltonian,
                                 alpha, beta, nu, use_bias=True):

    if use_bias:
        theta = parameters[:-1]
        bias = parameters[-1]
    else:
        theta = parameters
        bias = 0.0

    # =========================
    # Forward
    # =========================
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    term_count = hamiltonian.get_term_count()
    terms = [hamiltonian.get_term(i) for i in range(term_count)]

    EV_array = np.array([
        term.get_expectation_value(state).real
        for term in terms
    ])

    z = alpha * EV_array + bias
    tanh_vals = np.tanh(z)

    sech2 = 1.0 - tanh_vals**2
    d_tanh = alpha * sech2

    # =========================
    # dL / dE（完全一致）
    # =========================
    W_lower = np.tril(W, -1)

    interaction = (
        (W_lower @ tanh_vals) +
        (W_lower.T @ tanh_vals)
    ) * d_tanh

    node_term = h * d_tanh
    reg_term = beta * (2.0 * tanh_vals * d_tanh) / m

    dL_dEi = interaction + node_term + reg_term

    # =========================
    # Observable
    # =========================
    obs_all = Observable(n)

    for coef, term in zip(dL_dEi, terms):
        if abs(coef) > 1e-12:
            obs_all.add_operator(
                PauliOperator(term.get_pauli_string(), float(coef))
            )

    g_theta = np.asarray(ansatz.backprop(obs_all), dtype=float)

    # =========================
    # bias 勾配（完全一致）
    # =========================
    if use_bias:
        interaction_bias = np.sum(
            W_lower * (
                np.outer(sech2, tanh_vals) +
                np.outer(tanh_vals, sech2)
            )
        )

        node_bias = np.sum(h * sech2)
        reg_bias = np.sum(beta * (2.0 * tanh_vals * sech2) / m)

        dL_dbias = interaction_bias + node_bias + reg_bias

        return np.concatenate([g_theta, [dL_dbias]])

    return g_theta

def backprop_bias_y(parameters, n, m, ansatz, W, h, hamiltonian,
                                 alpha, beta, nu):

    theta = parameters[:-1]
    b = parameters[-1]

    # =========================
    # Forward
    # =========================
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    term_count = hamiltonian.get_term_count()
    terms = [hamiltonian.get_term(i) for i in range(term_count)]

    EV_array = np.array([
        term.get_expectation_value(state).real
        for term in terms
    ])

    # =========================
    # Nonlinear
    # =========================
    y_vals = np.tanh(alpha * EV_array)
    x_vals = np.tanh(alpha * EV_array + b)

    dx_dE = alpha * (1.0 - x_vals**2)
    dy_dE = alpha * (1.0 - y_vals**2)

    # =========================
    # dL / dE（完全一致）
    # =========================
    W_lower = np.tril(W, -1)

    interaction = (
        (W_lower @ x_vals) +
        (W_lower.T @ x_vals)
    ) * dx_dE

    linear_term = h * dx_dE

    reg_term = (2.0 * beta / m) * y_vals * dy_dE

    dL_dEi = interaction + linear_term + reg_term

    # =========================
    # Observable
    # =========================
    obs_all = Observable(n)

    for coef, term in zip(dL_dEi, terms):
        if abs(coef) > 1e-12:
            obs_all.add_operator(
                PauliOperator(term.get_pauli_string(), float(coef))
            )

    # =========================
    # θ gradient
    # =========================
    g_theta = np.asarray(ansatz.backprop(obs_all), dtype=float)

    # =========================
    # bias gradient（元コードと完全一致）
    # =========================
    dL_db = np.sum(
        ((np.dot(W, x_vals) + np.dot(W.T, x_vals)) + h) * (1.0 - x_vals**2)
    )

    return np.concatenate([g_theta, [dL_db]])

def backprop(parameters, n, m, ansatz, W, h, hamiltonian,
                          alpha, beta, nu):

    theta = parameters

    # =========================
    # Forward
    # =========================
    ansatz.set_parameter(theta)
    state = QuantumState(n)
    ansatz.update_quantum_state(state)

    term_count = hamiltonian.get_term_count()
    terms = [hamiltonian.get_term(i) for i in range(term_count)]

    EV_array = np.array([
        term.get_expectation_value(state).real
        for term in terms
    ])

    # =========================
    # Nonlinear（bias=0）
    # =========================
    y_vals = np.tanh(alpha * EV_array)
    x_vals = y_vals  # 明示的に一致

    dx_dE = alpha * (1.0 - x_vals**2)
    dy_dE = alpha * (1.0 - y_vals**2)

    # =========================
    # dL / dE（完全一致）
    # =========================
    W_lower = np.tril(W, -1)

    interaction = (
        (W_lower @ x_vals) +
        (W_lower.T @ x_vals)
    ) * dx_dE

    linear_term = h * dx_dE

    reg_term = (2.0 * beta / m) * y_vals * dy_dE

    dL_dEi = interaction + linear_term + reg_term

    # =========================
    # Observable
    # =========================
    obs_all = Observable(n)

    for coef, term in zip(dL_dEi, terms):
        if abs(coef) > 1e-12:
            obs_all.add_operator(
                PauliOperator(term.get_pauli_string(), float(coef))
            )

    # =========================
    # θ 勾配
    # =========================
    g_theta = np.asarray(ansatz.backprop(obs_all), dtype=float)

    return g_theta
