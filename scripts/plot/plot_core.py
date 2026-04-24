import matplotlib.pyplot as plt
from src.analysis.aggregator import aggregate

DEFAULT_BETAS = [-0.1, 0.0, 0.1, 0.2]

def compute_stats(vals):
    return (
        min(vals),
        max(vals),
        sum(vals) / len(vals)
    )

def plot_energy(energy_nb, energy_wb, target_betas, aggregation, save_path):
    plt.figure()

    DATASETS = [
        ("No bias", energy_nb, "no_reg",
         dict(color="black", marker="o", linestyle="-")),

        ("Bias x", energy_wb, "x",
         dict(color="blue", marker="s", linestyle="--")),

        ("Bias y", energy_wb, "y",
         dict(color="red", marker="^", linestyle=":")),
    ]

    for beta in target_betas:

        for label_prefix, data_dict, reg_type, style in DATASETS:

            if beta not in data_dict:
                continue

            xs, mean_vals, min_vals, max_vals = [], [], [], []

            for a in sorted(data_dict[beta].keys()):
                if reg_type not in data_dict[beta][a]:
                    continue

                vals = data_dict[beta][a][reg_type]

                xs.append(a)
                mean_vals.append(sum(vals)/len(vals))
                min_vals.append(min(vals))
                max_vals.append(max(vals))

            if not xs:
                continue

            # --- band ---
            if aggregation == "band":
                plt.fill_between(xs, min_vals, max_vals,
                                 color=style["color"], alpha=0.15)

                plt.plot(xs, mean_vals,
                         label=f"{label_prefix} (beta={beta})",
                         **style)

            # --- mean/min ---
            else:
                ys = []
                for a in sorted(data_dict[beta].keys()):
                    if reg_type not in data_dict[beta][a]:
                        continue
                    vals = data_dict[beta][a][reg_type]
                    ys.append(aggregate(vals, aggregation))

                plt.plot(xs, ys,
                         label=f"{label_prefix} (beta={beta})",
                         **style)
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()
