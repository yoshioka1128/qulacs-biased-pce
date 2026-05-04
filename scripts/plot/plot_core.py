import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.aggregator import aggregate
import pltutils

DEFAULT_BETAS = [-0.1, 0.0, 0.1, 0.2]

beta_marker_map = {
    -0.1: "o",
    0.0: "s",
    0.1: "^",
    0.2: "D",
}

def make_label(mode, bias_mode, label_prefix, beta, suffix=None):
    if mode == "method":
        base = f"{label_prefix}, beta={beta}"
    else:
        base = f"{bias_mode}, beta={beta}"

    return f"{base} {suffix}" if suffix else base


def compute_stats(vals):
    return (
        min(vals),
        max(vals),
        sum(vals) / len(vals)
    )


def build_datasets(
    mode,
    bias_mode,
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

        if bias_mode == "nobias":
            e_dict = cost_nb
            l_dict = loss_nb
            reg_type = "no_reg"

        elif bias_mode == "bias_x":
            e_dict = cost_wb
            l_dict = loss_wb
            reg_type = "x"

        elif bias_mode == "bias_y":
            e_dict = cost_wb
            l_dict = loss_wb
            reg_type = "y"

        else:
            raise ValueError(f"Unknown bias_mode: {bias_mode}")

        datasets = []

        for beta in target_betas:
            datasets.append(
                (
                    bias_mode,
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
        ax,
        cost_nb,
        cost_wb,
        target_betas,
        aggregation,
#        save_path,
        loss_nb=None,
        loss_wb=None,
        mode="method",
        bias_mode="nobias",
        show_label=True,
        add_legend=True,
        xmin=None,
        xmax=None,
#        close_fig=False,
):
#    fig, ax = plt.subplots()

    datasets = build_datasets(
        mode=mode,
        bias_mode=bias_mode,
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
            xs_orig = np.array(xs)
            color = style["color"]

            # =====================
            # cost
            # =====================

#            label = make_label(mode, bias_mode, label_prefix, beta)

            if aggregation == "band":
                mean_e = np.array(mean_e)
                min_e = np.array(min_e)
                max_e = np.array(max_e)

                xs_plot = xs_orig.copy()
                mean_plot = mean_e.copy()
                min_plot = min_e.copy()
                max_plot = max_e.copy()
                if xmin is not None or xmax is not None:
                    mask = np.ones_like(xs_plot, dtype=bool)

                    if xmin is not None:
                        mask &= xs_plot >= xmin
                    if xmax is not None:
                        mask &= xs_plot <= xmax

                    xs_plot = xs_plot[mask]
                    mean_plot = mean_plot[mask]
                    min_plot = min_plot[mask]
                    max_plot = max_plot[mask]

                ax.fill_between(
                    xs_plot,
                    min_plot,
                    max_plot,
                    color=color,
                    alpha=0.15,
                )

                ax.plot(
                    xs_plot,
                    mean_plot,
                    linestyle="-",
                    marker=beta_marker_map[beta],
                    color=color,
                    label=rf"$\beta ={beta}$",
                )

            else:
                mean_e = np.array(mean_e)
                min_e = np.array(min_e)
                max_e = np.array(max_e)
                if xmin is not None:
                    mask = xs >= xmin
                    xs = xs[mask]
                    mean_e = mean_e[mask]
                    min_e = min_e[mask]
                    max_e = max_e[mask]

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
                    bias_mode,
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
#    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)

    ax.grid(True, linestyle=":", linewidth=0.5)

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ax.legend(by_label.values(), by_label.keys())


def plot_stacked_bar_by_hour(
    ax,
    hours,
    df_pivot,
    consumer_order,
    consumer_color_dict,
    start_time,
    end_time,
    time_step,
):
    """
    Draw a stacked bar chart for each target hour using a given Axes.
    """
    for hour in hours:
        print("hour(plot):", hour)

        # Extract data for the current hour
        df_h = df_pivot[df_pivot["Hour"] == hour]

        # Reindex consumers to keep a fixed stacking order
        df_h = (
            df_h
            .set_index("Consumer")
            .reindex(consumer_order, fill_value=0.0)
            .reset_index()
        )

        # Skip if there is no data for this hour
        if df_h.empty:
            continue

        consumers = df_h["Consumer"].values

        # Convert kWh to MWh and apply the 10% factor
        values = df_h["Mean"].values / 10 / 1000

        # Assign colors per consumer
        colors = [consumer_color_dict[c] for c in consumers]

        # Draw stacked bars
        bottom = 0.0
        for v, c in zip(values, colors):
            ax.bar(
                hour,
                v,
                bottom=bottom,
                width=0.8,
                color=c,
                linewidth=0,
                edgecolor="none",
                rasterized=True,
            )
            bottom += v

def finalize_plot(
    fig,
    ax,
    hours,
    start_time,
    end_time,
    time_step,
    ylim_max=None,
):

    # Axis labels
    ax.set_xlabel("time $t$", fontsize = 24)
    ax.set_ylabel("Procured power $P_t$ [MWh]", fontsize = 24)

    # Axis ranges and ticks
    xmin = max(start_time - 4.5, 0.5)
    xmax = min(end_time+1 + 4.5, 24.5)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(range(int(np.ceil(xmin)), int(np.floor(xmax)) + 1))
    ax.set_ylim(0, ylim_max)
    ax.tick_params(axis='both', labelsize=16)

    # Tick appearance
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )

    # Grid lines
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Adjust figure layout
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.12)

def plot_negawatt_with_std(
    ax,
    hours,
    avg,
    std,
    ylim_max=None,
):
    """
    Plot negawatt average with std band and optional procurement line.
    """
    hours = np.asarray(hours)

    avg   = np.asarray(avg)
    std   = np.asarray(std)

    # 0 を「無効」とみなす
    valid = avg != 0

    # True が連続する区間を見つける
    indices = np.where(valid)[0]

    if len(indices) > 0:
        # 連続区間ごとに分割
        splits = np.where(np.diff(indices) != 1)[0] + 1
        segments = np.split(indices, splits)

        for i, seg in enumerate(segments):
            h = hours[seg]
            a = avg[seg]
            s = std[seg]

            # 線
            ax.plot(
                h,
                a,
                marker="o",
                color="blue",
                clip_on=False,
                label=r"$P_t$" if i == 0 else None,  # label は最初だけ
            )

            # 影
            ax.fill_between(
                h,
                a - s,
                a + s,
                color="blue",
                alpha=0.3,
                label=r"$\pm\sigma_t$" if i == 0 else None,
            )

    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)

    ax.grid(True)

def procurement_load_three_methods(rate):
    base = "outputs/power_opt/csv"

    file1 = f"{base}/procurement_pce_greedy_756nodes_rate{rate}_iseed42_start11_end20.csv"
    file2 = f"{base}/procurement_greedy_allzero_756nodes_rate{rate}_iseed42_start11_end20.csv"
    file3 = f"{base}/procurement_gurobi_756nodes_rate{rate}_iseed42_start11_end20.csv"

    return pd.read_csv(file1), pd.read_csv(file2), pd.read_csv(file3)


def procurement_compute_metrics(df):
    hours = df["hours"]
    diff = np.abs(df["total_means"] - df["proc"])
    ratio = df["total_std_means"] / df["total_means"]
    return hours, diff, ratio


def procurement_plot_one(ax_top, ax_bottom, dfs, show_ylabel=True, panel_labels=None):
    labels = ["pce simulation", "greedy (all-zero)", "optimal"]
    markers = ['o', 'x', 'D']
    colors = ['C0', 'green', 'C1']

    for df, label, m, color in zip(dfs, labels, markers, colors):
        h, d, r = procurement_compute_metrics(df)
        ax_top.plot(h, d*1000, marker=m, label=label, color=color, clip_on=False, zorder=3)
        ax_bottom.plot(h, r, marker=m, label=label, color=color, clip_on=False, zorder=3)

    if show_ylabel:
        ax_top.set_ylabel(r"deviation $|P_t - P_t^{\rm target}|$ [kWh]")
        ax_bottom.set_ylabel(r"coefficient of variation $\sigma_t / P_t$")

    ax_top.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax_bottom.grid(True, which="both", linestyle=":", linewidth=0.5)

    # =========================
    # (a)(b) ラベル
    # =========================
    if panel_labels is not None:
        top_label, bottom_label = panel_labels

        ax_top.text(
            0.02, 0.95, top_label,
            transform=ax_top.transAxes,
            va="top"
        )

        ax_bottom.text(
            0.02, 0.95, bottom_label,
            transform=ax_bottom.transAxes,
            va="top"
        )
    ax_top.set_xticks(range(11, 21))
    ax_top.set_ylim(bottom=0.0)
    ax_top.yaxis.set_label_coords(-0.11, 0.5)
    ax_bottom.set_xticks(range(11, 21))
    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.yaxis.set_label_coords(-0.11, 0.5)
