import numpy as np
import pandas as pd
from .demand import load_power_dataframe, build_proc_vector
from .covariance import load_covariance, build_index_map, extract_sub_covariance
from gurobi_energy_mathopt.data_loader import load_selected_originals, load_gurobi_result_row
from param_enemane.data_loader import BASE_DIR_PARAM

class PowerDataLoader:
    def __init__(self, nodes, iseed, rate, str_large=""):
        self.nodes = nodes
        self.iseed = iseed
        self.rate = rate
        self.str_large = str_large

        # --- caches ---
        self._proc = None
        self._power_dict = None
        self._cov_cache = {}
        self._idx_cache = {}

        # --- static load ---
        self._load_consumer_list()
        self._load_power()

    # =========================
    # init helpers
    # =========================
    def _load_consumer_list(self):
#        file = f"gurobi_energy_mathopt/output/selected_originals_L{self.nodes}_iseed{self.iseed}.csv"
        df = load_selected_originals(self.nodes, self.iseed)
        self.consumer_list_all = df["Consumer"].tolist()

    def _load_power(self):
        df = pd.read_csv(f"{BASE_DIR_PARAM}/param/power_consumption_hourly_mixup_restricted{self.str_large}.csv")
#        file = f"param-enemane/param/power_consumption_hourly_mixup_restricted{self.str_large}.csv"
#        df = pd.read_csv(file)

        self._power_dict = {
            h: g.set_index("Consumer")["Mean"]
            for h, g in df.groupby("Hour")
        }

    # =========================
    # public API
    # =========================

    def get_power(self, hour):
        """Pvec"""
        return self._power_dict[hour].loc[self.consumer_list_all].to_numpy()

    def get_proc(self):
        if self._proc is not None:
            return self._proc

        df = load_power_dataframe(self.nodes)

        self._proc = build_proc_vector(
            df,
            self.consumer_list_all,
            self.nodes,
            self.rate,
        )

        return self._proc

    def get_proc_at(self, hour):
        return self.get_proc()[hour - 1]

    def get_aligned_data(self, hour):
        # --- power ---
        P_series = self._power_dict[hour]
        Pvec = P_series.loc[self.consumer_list_all].to_numpy()

        # --- covariance ---
        cov_matrix, all_names = load_covariance(hour, self.str_large)

        name_to_idx = {name: i for i, name in enumerate(all_names)}

        missing = [c for c in self.consumer_list_all if c not in name_to_idx]
        if missing:
            raise ValueError(f"Missing in covariance: {missing[:5]} ...")

        idx = np.array([name_to_idx[c] for c in self.consumer_list_all])

        sub_cov = cov_matrix[np.ix_(idx, idx)]

        target = self.get_proc_at(hour)

        return sub_cov, Pvec, idx, target

