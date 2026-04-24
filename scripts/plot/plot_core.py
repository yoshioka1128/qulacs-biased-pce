import matplotlib.pyplot as plt
from src.analysis.aggregator import aggregate

DEFAULT_BETAS = [-0.1, 0.0, 0.1, 0.2]

def compute_stats(vals):
    return (
        min(vals),
        max(vals),
        sum(vals) / len(vals)
    )

def plot_energy(
    energy_nb,
    energy_wb,
    target_betas,
    aggregation,
    save_path,
    loss_nb=None,
    loss_wb=None,
    mode="method",   # ★追加
    model="no_bias",
):
    plt.figure()

    # =========================
    # DATASETS定義
    # =========================
    if mode == "method":
        DATASETS = [
            ("No bias", energy_nb, loss_nb, "no_reg",
             dict(color="black", marker="o")),

            ("Bias x", energy_wb, loss_wb, "x",
             dict(color="blue", marker="s")),

            ("Bias y", energy_wb, loss_wb, "y",
             dict(color="red", marker="^")),
        ]

    elif mode == "beta":
        beta_color_map = {
            -0.1: "C2",
            0.0: "C0",
            0.1: "C1",
            0.2: "C3",
        }

        if model == "no_bias":
            e_dict = energy_nb
            l_dict = loss_nb
            reg_type = "no_reg"

        elif model == "bias_x":
            e_dict = energy_wb
            l_dict = loss_wb
            reg_type = "x"

        elif model == "bias_y":
            e_dict = energy_wb
            l_dict = loss_wb
            reg_type = "y"

        else:
            raise ValueError(f"Unknown model: {model}")
        
        # no_biasのみ使う（必要なら拡張可能）
        DATASETS = []
        for beta in target_betas:
            DATASETS.append((
                f"{model}, beta={beta}",
                e_dict,
                l_dict,
                reg_type,
                dict(color=beta_color_map[beta], marker="o"),
                beta
            ))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # =========================
    # ループ
    # =========================
    for item in DATASETS:

        # modeで構造が違うので分岐
        if mode == "method":
            label_prefix, e_dict, l_dict, reg_type, style = item
            beta_loop = target_betas

        else:  # beta mode
            label_prefix, e_dict, l_dict, reg_type, style, fixed_beta = item
            beta_loop = [fixed_beta]

        for beta in beta_loop:

            if beta not in e_dict:
                continue

            xs = []
            mean_e, min_e, max_e = [], [], []
            mean_l = []

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
                    mean_l.append(sum(l_vals)/len(l_vals))

            if not xs:
                continue

            color = style["color"]

            # =====================
            # energy
            # =====================
            if aggregation == "band":
                plt.fill_between(xs, min_e, max_e,
                                 color=color, alpha=0.15)

                plt.plot(xs, mean_e,
                         linestyle="-",
                         marker=style["marker"],
                         color=color,
                         label=f"{model}, beta={beta} energy")

            else:
                ys = []
                for a in sorted(e_dict[beta].keys()):
                    if reg_type not in e_dict[beta][a]:
                        continue
                    vals = e_dict[beta][a][reg_type]
                    ys.append(aggregate(vals, aggregation))

                plt.plot(xs, ys,
                         linestyle="-",
                         marker=style["marker"],
                         color=color,
                         label=f"{model}, beta={beta} energy")

            # =====================
            # loss
            # =====================
            if l_dict is not None and mean_l:
                plt.plot(xs, mean_l,
                         linestyle="--",
                         color=color,
                         label=f"{model}, beta={beta} energy")

    plt.xlabel("alphasc")
    plt.ylabel("value")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
   
