import matplotlib.pyplot as plt
from src.analysis.aggregator import aggregate

DEFAULT_BETAS = [-0.1, 0.0, 0.1, 0.2]

beta_marker_map = {
    -0.1: "o",
    0.0: "s",
    0.1: "^",
    0.2: "D",
}

def make_label(mode, model, label_prefix, beta, suffix=None):
    if mode == "method":
        base = f"{label_prefix}, beta={beta}"
    else:
        base = f"{model}, beta={beta}"

    return f"{base} {suffix}" if suffix else base


def compute_stats(vals):
    return (
        min(vals),
        max(vals),
        sum(vals) / len(vals)
    )


def build_datasets(
    mode,
    model,
    cost_nb,
    cost_wb,
    loss_nb,
    loss_wb,
    target_betas,
):
    """
    plot用DATASETS生成
    """

    if mode == "method":
        return [
            (
                "No bias",
                cost_nb,
                loss_nb,
                "no_reg",
                dict(color="black"),
                target_betas,
            ),
            (
                "Bias x",
                cost_wb,
                loss_wb,
                "x",
                dict(color="blue"),
                target_betas,
            ),
            (
                "Bias y",
                cost_wb,
                loss_wb,
                "y",
                dict(color="red"),
                target_betas,
            ),
        ]

    elif mode == "beta":

        beta_color_map = {
            -0.1: "C2",
            0.0: "C0",
            0.1: "C1",
            0.2: "C3",
        }

        if model == "no_bias":
            e_dict = cost_nb
            l_dict = loss_nb
            reg_type = "no_reg"

        elif model == "bias_x":
            e_dict = cost_wb
            l_dict = loss_wb
            reg_type = "x"

        elif model == "bias_y":
            e_dict = cost_wb
            l_dict = loss_wb
            reg_type = "y"

        else:
            raise ValueError(f"Unknown model: {model}")

        datasets = []

        for beta in target_betas:
            datasets.append(
                (
                    model,
                    e_dict,
                    l_dict,
                    reg_type,
                    dict(color=beta_color_map[beta], marker="o"),
                    [beta],
                )
            )

        return datasets

    else:
        raise ValueError(f"Unknown mode: {mode}")


def collect_series(e_dict, l_dict, beta, reg_type):
    xs = []
    mean_e, min_e, max_e = [], [], []
    mean_l = []

    if beta not in e_dict:
        return None

    for a in sorted(e_dict[beta].keys()):

        if reg_type not in e_dict[beta][a]:
            continue

        e_vals = e_dict[beta][a][reg_type]
        mn, mx, avg = compute_stats(e_vals)

        xs.append(a)
        mean_e.append(avg)
        min_e.append(mn)
        max_e.append(mx)

        if l_dict is not None:
            l_vals = l_dict[beta][a][reg_type]
            mean_l.append(sum(l_vals) / len(l_vals))

    if not xs:
        return None

    return xs, mean_e, min_e, max_e, mean_l


def plot_cost(
        cost_nb,
        cost_wb,
        target_betas,
        aggregation,
        save_path,
        loss_nb=None,
        loss_wb=None,
        mode="method",
        model="no_bias",
        close_fig=False,
):

    fig, ax = plt.subplots()

    datasets = build_datasets(
        mode=mode,
        model=model,
        cost_nb=cost_nb,
        cost_wb=cost_wb,
        loss_nb=loss_nb,
        loss_wb=loss_wb,
        target_betas=target_betas,
    )

    for label_prefix, e_dict, l_dict, reg_type, style, beta_loop in datasets:

        for beta in beta_loop:

            result = collect_series(
                e_dict=e_dict,
                l_dict=l_dict,
                beta=beta,
                reg_type=reg_type,
            )

            if result is None:
                continue

            xs, mean_e, min_e, max_e, mean_l = result

            color = style["color"]

            # =====================
            # cost
            # =====================

            label = make_label(mode, model, label_prefix, beta)

            if aggregation == "band":

                ax.fill_between(
                    xs,
                    min_e,
                    max_e,
                    color=color,
                    alpha=0.15,
                )

                ax.plot(
                    xs,
                    mean_e,
                    linestyle="-",
                    marker=beta_marker_map[beta],
                    color=color,
                    label=label,
                )

            else:

                ys = []

                for a in sorted(e_dict[beta].keys()):

                    if reg_type not in e_dict[beta][a]:
                        continue

                    vals = e_dict[beta][a][reg_type]
                    ys.append(aggregate(vals, aggregation))

                ax.plot(
                    xs,
                    ys,
                    linestyle="-",
                    marker=beta_marker_map[beta],
                    color=color,
                    label=label,
                )

            # =====================
            # loss
            # =====================

            if l_dict is not None and mean_l:

                loss_label = make_label(
                    mode,
                    model,
                    label_prefix,
                    beta,
                    suffix="loss",
                )

                ax.plot(
                    xs,
                    mean_l,
                    linestyle="--",
                    color=color,
                    label=loss_label,
                )

    # =====================
    # axis settings
    # =====================

    ax.set_xlabel("alphasc")
    ax.set_ylabel("value")

    ax.set_yscale("log")
#    ax.set_ylim(1.0e-5, 0.02)
    ax.set_ylim(top=0.02)
    ax.set_xlim(0.0, 3.0)

    ax.grid(True)

    # 凡例重複除去
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys())

    fig.tight_layout()
    fig.savefig(save_path)
    if close_fig:
        plt.close(fig)
