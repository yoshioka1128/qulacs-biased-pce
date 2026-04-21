# src/domain/loss.py
import numpy as np
from qulacs import QuantumState

# =========================
# Core loss
# =========================
def compute_loss(J, h, n_qubits, theta, ansatz, hamiltonian, alpha, beta, bias=None):
    exp_value = _compute_expectation(n_qubits, theta, ansatz, hamiltonian)
    z = alpha * exp_value + (bias if bias is not None else 0.0)
    x = np.tanh(z)
    y = np.tanh(alpha * exp_value)
    energy = _compute_energy(J, h, x, y, beta)
    return energy, exp_value


def power_loss_func_bias(J, h, n_qubits, para, bias, ansatz, hamiltonian, alpha, beta):
    exp_value = _compute_expectation(n_qubits, para, ansatz, hamiltonian)

    x = np.tanh(alpha * exp_value + bias)

    return _compute_energy(J, h, x, beta), exp_value

def power_loss_func_fast(J, h, n_qubits, para, ansatz, hamiltonian, alpha, beta):
    exp_value = _compute_expectation(n_qubits, para, ansatz, hamiltonian)

    x = np.tanh(alpha * exp_value)

    return _compute_energy(J, h, x, beta), exp_value

def _compute_energy(J, h, x, y, beta):
    n_nodes = len(h)

    interaction = np.sum(np.tril(J, -1) * np.outer(x, x))
    node = np.dot(h, x)
    reg = beta * np.mean(y**2)

    return interaction + node + reg

def _compute_expectation(n_qubits, para, ansatz, hamiltonian):
    ansatz.set_parameter(para)

    state = QuantumState(n_qubits)
    ansatz.update_quantum_state(state)

    return np.array([
        hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])

