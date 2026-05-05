# aggregator.py
import os, re
import numpy as np
import pandas as pd
from param_enemane.data_loader import BASE_DIR_PARAM
from gurobi_energy_mathopt.data_loader import BASE_DIR_GUROBI

def evaluate_solution(hour, solution, loader):
    sol = (np.array(solution) != 1).astype(np.float64)

    Pvec = loader.get_power(hour)
    cov  = loader.get_cov(hour)
    target = loader.get_proc_at(hour)

    var = (sol @ cov @ sol) / (1000 * 10)**2
    Ptot = sol @ Pvec
    dev = np.abs(Ptot / 1000 / 10 - target)

    return var, dev, Ptot

def process_covariance_file(
    file,
    df,
    consumer_list,
):
    filename = os.path.basename(file)
    match = re.search(r"time(\d+)_", filename)
    if not match:
        return None
    hour = int(match.group(1))
    print('hour: ', hour)

    with np.load(file, allow_pickle=True) as data:
        cov_matrix = data["cov"]        # ndarray（float32）
        all_names = data["names"].tolist()  # list[str]
    cov_df = pd.DataFrame(cov_matrix, index=all_names, columns=all_names)

    original_cols = [c for c in cov_df.columns if c in consumer_list]
    cov_df_original = cov_df.loc[original_cols, original_cols]

    df_hour_original = df[(df["Hour"] == hour) & (df["Consumer"].isin(original_cols))]

#    check_variance_match_single(
#        hour, cov_df_original, df_hour_original, original_cols
#    )

    cov_matrix = cov_df_original.to_numpy()
    avg_total = float(np.sum(df_hour_original["Mean"].to_numpy(), dtype=np.float64))
    var_total = float(np.sum(cov_matrix, dtype=np.float64))
    std_total = np.sqrt(max(var_total, 0.0))

    stats = {"Hour": hour, "AvgTotal": avg_total, "StdTotal": std_total,}

    return stats

def compute_total_stats(
    df,
    consumer_lists,
    hours,
    str_large=""
):
    stats_total_list = []

    for it, hour in enumerate(hours):
        file = f"{BASE_DIR_PARAM}/param/covariance_matrix_time{hour:02d}_mixup_restricted{str_large}.npz"
        result = process_covariance_file(file, df, consumer_lists[it],)
        if result is None:
            continue

        stats = result
        stats_total_list.append(stats)

    return stats_total_list

def proc_from_mean(nodes, iseed, str_large):
    input_file = f"{BASE_DIR_PARAM}/param/power_consumption_hourly_mixup_restricted{str_large}.csv"
    df = pd.read_csv(input_file)
    consumer_list_file = f"{BASE_DIR_GUROBI}/output/selected_originals_L{nodes}_iseed{iseed}.csv"
    consumer_L_list = pd.read_csv(consumer_list_file, header=None).iloc[1:, 0].tolist()
    df_L = df[df["Consumer"].isin(consumer_L_list)]
    mean_by_hour = df_L.groupby("Hour")["Mean"].mean()

    proc = mean_by_hour.reindex(range(1, 25)).to_numpy()
    return proc
