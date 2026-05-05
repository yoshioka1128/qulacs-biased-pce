import pandas as pd
import numpy as np


def load_power_dataframe(str_large=""):
    file = f"param-enemane/param/power_consumption_hourly_mixup_restricted{str_large}.csv"
    return pd.read_csv(file)


def build_proc_vector(df_power, consumer_list, nodes, rate):
    """
    proc vector (length=24)
    """

    df_L = df_power[df_power["Consumer"].isin(consumer_list)]
    mean_by_hour = df_L.groupby("Hour")["Mean"].mean()

    proc = mean_by_hour.reindex(range(1, 25)).to_numpy()

    # scaling（legacy踏襲）
    proc = proc * nodes * rate / 1000 / 10

    return proc
