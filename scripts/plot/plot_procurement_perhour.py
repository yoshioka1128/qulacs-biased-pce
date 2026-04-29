# plot_pce_procured_perhour.py
import pandas as pd
import numpy as np
from core.graph_handler import prepare_int
import glob, re, os, json, csv
import matplotlib.pyplot as plt
import pltutils
#import pltutils_3fig

from src.core.utils import make_consumer_color_dict
from src.analysis.aggregator import compute_total_stats
from src.analysis.aggregator import proc_from_mean

from scripts.plot.plot_core import plot_stacked_bar_by_hour, finalize_plot, plot_negawatt_with_std

from src.config.node_config import NODE_CONFIG
from gurobi_energy_mathopt.data_loader import load_selected_originals, BASE_DIR_GUROBI
from param_enemane.data_loader import BASE_DIR_PARAM

algo = str(input('algo (pce): ') or 'pce')
choice = input('Greedy method y/n ? (y):') or 'y'
greedy = (choice=='y')
if greedy: str_greedy="greedy_"
else: str_greedy = ""

mode = str(input('mode (bias_y): ') or 'bias_y')
if mode == "nobias": str_mode = ""
else: str_mode = f"{mode}_"

nodes = int(input('node数を入力してください (756): ') or 756)
rate = float(input('rate (0.1): ') or 0.1)

node_cfg = NODE_CONFIG[nodes, 0.1, mode]
n_qubits = node_cfg.n_qubits
k = node_cfg.k
depth = node_cfg.depth
ninit = node_cfg.ninit
init = node_cfg.iinit
alphasc = node_cfg.alphasc
beta = node_cfg.beta
iseed = node_cfg.iseed

str_large=""
if nodes == 10296: str_large="_large"

start_time, end_time, time_step = 11, 20, 1

# --- Step 1: consumer_list.csv と results.json を読み込む ---
# consumer ID のリストを読み込む
df_selected_originals = load_selected_originals(nodes, iseed)
consumer_list = BASE_DIR_GUROBI / f"output/selected_originals_L{nodes}_iseed{iseed}.csv"
consumer_color_dict, consumer_list_all = make_consumer_color_dict(consumer_list)

time_list = list(range(start_time, end_time+1, time_step))
consumer_lists = []
# read results
if algo == 'pce':
    for time in time_list:
        output_pce = f"outputs/power_opt/time{time}_nT1_rate{rate}_{nodes}nodes_{n_qubits}qubits_{k}body_ninit{ninit}_depth{depth}_all2all_methodBFGS_iseed{iseed}/read"
        results_ws_json_file = f"{output_pce}/pce_{str_greedy}time_resolved_it{time}_results_backprop_{str_mode}alphasc{alphasc}_beta{beta}_init0.json"
        with open(results_ws_json_file, "r") as f:
            results_ws = json.load(f)
        solution_ws = results_ws["Solution for Minimum Energy"]
        consumer_lists.append([cid for cid, val in zip(consumer_list_all, solution_ws) if val == -1])
if algo == 'greedy':
    for time in time_list:
        output_pce = f"outputs/power_opt/time{time}_nT1_rate{rate}_{nodes}nodes_{n_qubits}qubits_{k}body_ninit{ninit}_depth{depth}_all2all_methodBFGS_iseed{iseed}/read"
        results_ws_json_file = f"{output_pce}/greedy_allzero_time_resolved_it{time}_results.json"
        with open(results_ws_json_file, "r") as f:
            results_ws = json.load(f)
        solution_ws = results_ws["Solution for Minimum Energy"]
        consumer_lists.append([cid for cid, val in zip(consumer_list_all, solution_ws) if val == -1])

# --- Step 2: 電力消費データを読み込んで対象 consumer のみ抽出 ---
df_power = pd.read_csv(f"{BASE_DIR_PARAM}/param/power_consumption_hourly_mixup_restricted{str_large}.csv")
for i, hour in enumerate(time_list):
    consumer_list = consumer_lists[i]
    df_h = df_power[
        (df_power["Hour"] == hour) &
        (df_power["Consumer"].isin(consumer_list))
    ]
    
# ===== プロット0 =====
# make procurement of negawatt
proc = proc_from_mean(nodes, iseed, str_large)
proc = proc * nodes * rate / 1000 /10
max_proc = np.max(proc)
ylim_max = max_proc*1.4

# make hours list
df_power = df_power[df_power["Hour"].between(start_time, end_time)]
one_consumer = df_power["Consumer"].iloc[0]
hours = (
    df_power[df_power["Consumer"] == one_consumer]["Hour"]
    .sort_values()
    .to_numpy()
)

# --- Add individual consumer lines ---
df_pivot = []
for it, hour in enumerate(hours):
    consumer_list = consumer_lists[it]
    print('hour(pivot): ', hour)
    df_h = df_power[
        (df_power["Hour"] == hour) &
        (df_power["Consumer"].isin(consumer_list))
    ][["Consumer", "Mean"]]

    df_h["Hour"] = hour
    df_pivot.append(df_h)

df_pivot = pd.concat(df_pivot)

# set proc
proc_arr = np.asarray(proc)
all_hours = np.arange(1, len(proc_arr) + 1)
mask = (all_hours >= start_time) & (all_hours <= end_time)
proc = np.where(mask, proc_arr, 0.0)

# get covariance and negawatt
stats_list_ws = compute_total_stats(df_power, consumer_lists, hours, str_large)
for stat in stats_list_ws:
    print('warm start', stat)
stats_df_ws = pd.DataFrame(stats_list_ws).sort_values("Hour")
#stats_df_ws.to_csv(f"{output_pce}/pce_{str_greedy}negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}.csv", index=False)
# ===== プロット2 =====
avg_totals = stats_df_ws["AvgTotal"].to_numpy()
std_totals = stats_df_ws["StdTotal"].to_numpy()
reduced_avg_totals = avg_totals / 1000 / 10
reduced_std_totals = std_totals / 1000 / 10

output_df_ws = pd.DataFrame({
    'hours': hours,
    'total_means': reduced_avg_totals,
    'total_std_means': reduced_std_totals,
    'proc': proc[mask]
})

# =========================================
# algo に応じて共通文字列を切り替える
# =========================================

if algo == "pce":
    file_suffix = (
        f"_pce_{str_greedy}{str_mode}"
        f"{nodes}nodes_rate{rate}_iseed{iseed}"
    )

elif algo == "greedy":
    file_suffix = "_greedy_allzero"

else:
    raise ValueError(f"unknown algo: {algo}")

output_df_ws.to_csv(f"outputs/power_opt/csv/procurement{file_suffix}.csv", index=False)

fig, ax = plt.subplots()
plot_stacked_bar_by_hour(ax, hours, df_pivot, consumer_list_all, consumer_color_dict, start_time, end_time, time_step,)
plot_negawatt_with_std(ax, hours, reduced_avg_totals, reduced_std_totals, ylim_max=ylim_max,)
#ax.plot(hours, proc, linestyle="--", label="target", color="black",)
ax.plot(all_hours, proc, linestyle="--", label=r"$P^{\rm target}_t$", color="black",)
finalize_plot(fig, ax, hours, start_time, end_time, time_step, ylim_max=ylim_max,)
fig.savefig(f"outputs/power_opt/figures/png/procurement{file_suffix}_start{start_time}_end{end_time}.png")
ax.legend(loc="upper right", fontsize=16)
label = ax.text(0.02, 0.97, f"$m$ = {nodes}", transform=ax.transAxes, fontweight="bold", va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
fig.savefig(f"outputs/power_opt/figures/pdf/procurement{file_suffix}_start{start_time}_end{end_time}.pdf")
label.set_text("(b) PCE simulation")
label.set_fontsize(24)
fig.savefig(f"outputs/power_opt/figures/pdf/procurement_stack{file_suffix}_start{start_time}_end{end_time}_3fig.pdf")
fig.savefig(f"outputs/power_opt/figures/png/procurement_stack{file_suffix}_start{start_time}_end{end_time}_3fig.png")

fig, ax = plt.subplots()
plot_negawatt_with_std(ax, hours, reduced_avg_totals, reduced_std_totals, ylim_max=ylim_max,)
#ax.plot(hours, proc, linestyle="--", label="target", color="black",)
ax.plot(all_hours, proc, linestyle="--", label=r"$P^{\rm target}_t$", color="black",)
finalize_plot(fig, ax, hours, start_time, end_time, time_step, ylim_max=ylim_max,)
fig.savefig(f"outputs/power_opt/figures/png/procurement{file_suffix}_start{start_time}_end{end_time}.png")
ax.legend(loc="upper right", fontsize=16)
fig.savefig(f"outputs/power_opt/figures/pdf/procurement{file_suffix}_start{start_time}_end{end_time}.pdf")
fig.savefig(f"outputs/power_opt/figures/pdf/procurement{file_suffix}_start{start_time}_end{end_time}_3fig.pdf")
fig.savefig(f"outputs/power_opt/figures/png/procurement{file_suffix}_start{start_time}_end{end_time}_3fig.png")
plt.show()
