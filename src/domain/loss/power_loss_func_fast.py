# src/domain/loss.py

import numpy as np
from qulacs import QuantumState

def power_loss_func_bias(J, h, n_qubits, para, bias, ansatz, hamiltonian, alpha, beta):
    exp_value = _compute_expectation(n_qubits, para, ansatz, hamiltonian)

    x = np.tanh(alpha * exp_value + bias)

    return _compute_energy(J, h, x, beta), exp_value

def power_loss_func_fast(J, h, n_qubits, para, ansatz, hamiltonian, alpha, beta):
    exp_value = _compute_expectation(n_qubits, para, ansatz, hamiltonian)

    x = np.tanh(alpha * exp_value)

    return _compute_energy(J, h, x, beta), exp_value

def _compute_energy(J, h, x, beta):
    n_nodes = len(h)

    interaction = np.sum(np.tril(J, -1) * np.outer(x, x))
    node = np.dot(h, x)
    reg = beta * np.mean(x**2)

    return interaction + node + reg

def _compute_expectation(n_qubits, para, ansatz, hamiltonian):
    ansatz.set_parameter(para)

    state = QuantumState(n_qubits)
    ansatz.update_quantum_state(state)

    return np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

