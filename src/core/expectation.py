# expectation.py
import numpy as np
from qulacs import QuantumState

def compute_expectation(n_qubits, para, ansatz, hamiltonian):
    ansatz.set_parameter(para)

    state = QuantumState(n_qubits)
    ansatz.update_quantum_state(state)

    return np.array([
	hamiltonian.get_term(i).get_expectation_value(state).real
        for i in range(hamiltonian.get_term_count())
    ])
