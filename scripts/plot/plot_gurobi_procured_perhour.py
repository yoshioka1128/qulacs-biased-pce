import pandas as pd
import numpy as np
import glob, re, os, json, csv
import matplotlib.pyplot as plt

import pltutils
import ast
from pathlib import Path
from utils import make_consumer_color_dict
from analysis import compute_total_stats, proc_from_mean
from plots import plot_stacked_bar_by_hour, finalize_plot, plot_negawatt_with_std

nodes = int(input('node数を入力してください (10296): ') or 10296)
iseed = 42
rate = 0.5
start_time, end_time, time_step = 11, 20, 1
str_large=""
if nodes == 10296: str_large="_large"

# --- Step 1: consumer_list.csv と results.json を読み込む ---
consumer_list_file = f"gurobi_energy_mathopt/output/selected_originals_L{nodes}_iseed{iseed}.csv"
consumer_color_dict, consumer_list_all = make_consumer_color_dict(consumer_list_file)

# consumer ID のリストを読み込む
consumer_df = pd.read_csv(consumer_list_file)
consumer_list_all = consumer_df["Consumer"].tolist()

time_list = list(range(start_time, end_time+1, time_step))
selected_indices_csv = f"gurobi_energy_mathopt/output/results_nT1_L{nodes}_MNone_rate{rate}_iseed{42}_new.csv"
csv_df = pd.read_csv(selected_indices_csv)

consumer_lists = []
for time in time_list:
    row = csv_df.loc[csv_df["hour"] == time]
    if row.empty:
        consumer_lists.append([])
        continue
    indices = ast.literal_eval(row.iloc[0]["selected_indices"])
    consumer_lists.append([consumer_list_all[i] for i in indices])

# --- Step 2: 電力消費データを読み込んで対象 consumer のみ抽出 ---
df_power = pd.read_csv(f"param-enemane/param/power_consumption_hourly_mixup_restricted{str_large}.csv")
for hour, consumers in enumerate(consumer_lists):
    df_h = df_power[
        (df_power["Hour"] == hour) &
        (df_power["Consumer"].isin(consumers))
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
output_gurobi = f"outputs/power_opt/gurobi_optimized/nT1_rate{rate}_{nodes}nodes"
Path(output_gurobi).mkdir(parents=True, exist_ok=True)

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

fig, ax = plt.subplots()
plot_stacked_bar_by_hour(ax, hours, df_pivot, consumer_list_all, consumer_color_dict, start_time, end_time, time_step,)
#ax.plot(hours, proc, label="target", color="black",)
ax.plot(all_hours, proc, label=r"$P^{\rm target}_t$", color="black",)
finalize_plot(fig, ax, hours, start_time, end_time, time_step, ylim_max,)
fig.savefig(f"{output_gurobi}/gurobi_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.png")
ax.legend(loc="upper right", fontsize=16)
label = ax.text(0.02, 0.97, f"$m$ = {nodes}", transform=ax.transAxes, fontweight="bold", va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
fig.savefig(f"{output_gurobi}/gurobi_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.pdf")
import pltutils_3fig
if nodes == 10296: label.set_text("(c) optimal and suboptimal")
else: label.set_text("(c) optimal")
label.set_fontsize(24)
fig.savefig(f"{output_gurobi}/gurobi_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.pdf")
fig.savefig(f"{output_gurobi}/gurobi_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.png")
import pltutils
plt.show()

# get covariance and negawatt
stats_total_list = compute_total_stats(df_power, consumer_lists, hours, str_large)
for stat in stats_total_list:
    print(stat)
stats_df = pd.DataFrame(stats_total_list).sort_values("Hour")
stats_df.to_csv(f"{output_gurobi}/gurobi_negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}.csv", index=False)

# ===== プロット2 =====
avg_totals = stats_df["AvgTotal"].to_numpy()
std_totals = stats_df["StdTotal"].to_numpy()
reduced_avg_totals = avg_totals / 1000 / 10
reduced_std_totals = std_totals / 1000 / 10

output_df = pd.DataFrame({
    'hours': hours,
    'total_means': reduced_avg_totals,
    'total_std_means': reduced_std_totals,
    'proc': proc[mask]
})
output_df.to_csv(f"{output_gurobi}/gurobi_reduced_std_{nodes}nodes_rate{rate}_iseed{iseed}.csv", index=False)

fig, ax = plt.subplots()
plot_stacked_bar_by_hour(ax, hours, df_pivot, consumer_list_all, consumer_color_dict, start_time, end_time, time_step,)
plot_negawatt_with_std(ax, hours, reduced_avg_totals, reduced_std_totals, ylim_max=ylim_max,)
#ax.plot(hours, proc, linestyle="--", label="target", color="black",)
ax.plot(all_hours, proc, linestyle="--", label=r"$P^{\rm target}_t$", color="black",)
finalize_plot(fig, ax, hours, start_time, end_time, time_step, ylim_max=ylim_max,)
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.png")
ax.legend(loc="upper right", fontsize=16)
label = ax.text(0.02, 0.97, f"$m$ = {nodes}", transform=ax.transAxes, fontweight="bold", va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.pdf")
import pltutils_3fig
if nodes == 10296: label.set_text("(c) optimal and suboptimal")
else: label.set_text("(c) optimal")
label.set_fontsize(24)
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.pdf")
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_stack_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.png")
import pltutils
plt.show()

fig, ax = plt.subplots()
plot_negawatt_with_std(ax, hours, reduced_avg_totals, reduced_std_totals, ylim_max=ylim_max,)
#ax.plot(hours, proc, linestyle="--", label="target", color="black",)
ax.plot(all_hours, proc, linestyle="--", label=r"$P^{\rm target}_t$", color="black",)
finalize_plot(fig, ax, hours, start_time, end_time, time_step, ylim_max=ylim_max,)
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.png")
ax.legend(loc="upper right", fontsize=16)
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}.pdf")
import pltutils_3fig
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.pdf")
fig.savefig(f"{output_gurobi}/gurobi_negawatt_std_{nodes}nodes_rate{rate}_iseed{iseed}_start{start_time}_end{end_time}_3fig.png")
plt.show()

