# power_loss.py
import numpy as np
from qulacs import QuantumState
from src.core.expectation import compute_expectation

# =========================
# Core loss
# =========================
def compute_loss(
        J, h, n_qubits, theta, ansatz, hamiltonian,
        alpha, beta
):
    exp_value = compute_expectation(n_qubits, theta, ansatz, hamiltonian)
    z = alpha * exp_value 
    x = np.tanh(z)
    
    energy = _compute_energy(J, h, x, beta)
    return energy, exp_value

def compute_loss_bias_xy(
        J, h, n_qubits, theta, ansatz, hamiltonian,
        alpha, beta, bias=None, reg_type='y'
):
    exp_value = compute_expectation(n_qubits, theta, ansatz, hamiltonian)
    z = alpha * exp_value + (bias if bias is not None else 0.0)
    x = np.tanh(z)
    y = np.tanh(alpha * exp_value)
    
    energy = _compute_energy_xy(J, h, x, y, beta, reg_type)
    return energy, exp_value

def compute_loss_bias_x(
        J, h, n_qubits, theta, ansatz, hamiltonian,
        alpha, beta, bias
):
    exp_value = compute_expectation(n_qubits, theta, ansatz, hamiltonian)
    z = alpha * exp_value + bias
    x = np.tanh(z)
    
    energy = _compute_energy(J, h, x, beta)
    return energy, exp_value

def _compute_energy(J, h, x, beta):
    n_nodes = len(h)

    interaction = np.sum(np.tril(J, -1) * np.outer(x, x))
    node = np.dot(h, x)
    reg_func = beta * np.mean(x**2)

    return interaction + node + reg_func

def _compute_energy_xy(J, h, x, y, beta, reg_type="y"):

    n_nodes = len(h)

    interaction = np.sum(np.tril(J, -1) * np.outer(x, x))
    node = np.dot(h, x)
    reg_func = beta * np.mean(y**2)

    return interaction + node + reg_func

