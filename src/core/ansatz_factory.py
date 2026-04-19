from pce import BrickWorkAnsatz, ChainBrickWorkAnsatz, All2All

def select_ansatz(type_ansatz, n_qubits, depth):
    if type_ansatz == '1d_chain':
        return ChainBrickWorkAnsatz(n_qubits, depth)
    elif type_ansatz == '1d_brick':
        return BrickWorkAnsatz(n_qubits, depth)
    elif type_ansatz == 'all2all':
        return All2All(n_qubits, depth)
    else:
        raise ValueError(f"Unknown ansatz type: {type_ansatz}")
