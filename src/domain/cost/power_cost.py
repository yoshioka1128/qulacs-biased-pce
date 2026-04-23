# power_cost.py
def compute_cost(J, h, spin_config):
    J_sym = J + J.T
    interaction_term = 0.5 * float(spin_config @ J_sym @ spin_config)

    node_term = float(spin_config @ h)

    energy = interaction_term + node_term
    return energy
