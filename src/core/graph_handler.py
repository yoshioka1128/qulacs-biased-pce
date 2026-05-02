from graph_utils import setup_graph
from power_opt import param_PCE, read_param
from power import sk_cost_func
from src.core.utils import prepare_int_from_d, qubo2ising_correct
from param_enemane.data_loader import load_power_data, load_covariance_matrix
#from gurobi_energy_mathopt.data_loader import load_selected_originals
import pandas as pd
import gurobipy as gp
import gurobi_energy_mathopt.preprocessing as pre
import gurobi_energy_mathopt.utils as gurobi_utils
import numpy as np

def qubo_cost(w, sigma, p, proc, nT):
    L = len(w)
    mse_term = np.mean([(np.dot(p[:,t], w) - proc[t])**2 for t in range(nT)])
    quad_term = np.sum(sigma * np.outer(w, w))
    return quad_term + mse_term

def qubo_cost_from_x(x, consumer_list, hour, nT, rate, large_str):
    USE_NEW = True
    m = len(x)
    w = ((1 - np.array(x)) / 2).astype(int)

    sigma, Pt_ex, proc = gurobi_utils.load_covariance_and_expectation(USE_NEW, consumer_list, hour, nT, rate, large_str)
    L = len(w)
    residual = Pt_ex.T @ w - proc
    mse_term = float(residual @ residual / nT)
    mean = float(w @ Pt_ex.mean(axis=1))
    quad_term = float(w @ sigma @ w)

    return quad_term + mse_term, np.abs(mse_term), quad_term, mean

def model_solution_to_decimal(model, L, lsb_at_index0=True):
    """model.__data に {i: Var} として入っている前提。
    lsb_at_index0=True のとき、ビット i は 2**i の重み（LSB=index0）。"""
    if getattr(model, "SolCount", 0) <= 0:
        raise RuntimeError("モデルに解がありません。先に optimize() を実行してください。")

    w = model.__data  # {i: Var}
    bits = np.array([int(round(w[i].X)) for i in range(L)], dtype=int)

    if lsb_at_index0:
        # number = Σ bits[i] * 2**i
        number = int(np.sum(bits * (1 << np.arange(L))))
    else:
        # bits[0]をMSBとみなす場合はこちら（必要なら切替）
        number = int("".join(str(b) for b in bits), 2)

    return number, bits

def get_random_consumers(L, rng):
    df = load_power_data(L)
    if L <= 2143:
        consumers = df[df['Consumer'].str.startswith('Original')]['Consumer'].unique().tolist()
#    elif L == 10296:
#        input_file = f"./param-enemane/param/power_consumption_hourly_mixup_restricted_large.csv"
#        df = pd.read_csv(input_file)
#        consumers = df['Consumer'].unique().tolist()
    else:
        consumers = df['Consumer'].unique().tolist()
    consumers_list = rng.choice(consumers, size=L, replace=False)
    return consumers_list

def prepare_int(graph_type, m, it, nT, rate, rng, USE_NEW, iseed=42,
                consumer_list=None,
                gurobi_result=None,
                sigma=None, Pt_ex=None, proc=None):
    _ = get_random_consumers(m, rng).tolist() # tempolary added for consistency of rng 

    Cmin = None
    Cmax = None
    if graph_type != "power_opt":
        raise ValueError

    # backward compatibility
#    if consumer_list is None:
#        consumer_list = get_random_consumers(m, rng).tolist()

    if gurobi_result is not None:
        row = gurobi_result.loc[gurobi_result["hour"] == it]
        Cmin = row["obj_val_min"].iloc[0]
        Cmax = row["obj_val_max"].iloc[0]
        frob_norm = row["frobenius_norm"].iloc[0]

    frob_norm, shift, dJ, dhex = prepare_int_from_d(
        consumer_list=consumer_list,
        m=m,
        it=it,
        nT=nT,
        rate=rate
    )

#    else:
#        frob_norm, shift, dJ, dhex = prepare_int_from_d(
#            consumer_list=consumer_list,
#            m=m,
#            it=it,
#            nT=nT,
#            rate=rate
#        )
#
#        sigma, Pt_ex, proc = gurobi_utils.load_covariance_and_expectation(
#            USE_NEW, consumer_list, it, nT, rate
#        )
#
#        model = gurobi_utils.electric_power_demand_portfolio(
#            m, sigma, Pt_ex, proc, None, nT, False, sense=gp.GRB.MINIMIZE
#        )
#        model.optimize()
#
#        model_max = gurobi_utils.electric_power_demand_portfolio(
#            m, sigma, Pt_ex, proc, None, nT, False, sense=gp.GRB.MAXIMIZE
#        )
#        model_max.optimize()
#
#        Cmin = model.ObjVal
#        Cmax = model_max.ObjVal

    return dJ, dhex, Cmin, Cmax, frob_norm, shift, consumer_list
#def prepare_int(graph_type, m, it, nT, rate, rng, USE_NEW, iseed=42):
#    if graph_type == "power_opt":
#        if USE_NEW:
#            consumer_list = get_random_consumers(m, rng).tolist()
#            if iseed == 42:
#                df_selected_originals = load_selected_originals(m, iseed)
#                consumer_list = df_selected_originals["Consumer"].tolist()
#                frob_norm, shift, dJ, dhex = prepare_int_from_d(consumer_list, m, it, nT, rate)
#                result_file = f'gurobi_energy_mathopt/output/results_nT{nT}_L{m}_MNone_rate{rate}_iseed{iseed}_new.csv'
##                if it in [15, 19, 20] and m == 10296:
##                    result_file = f'gurobi_energy_mathopt/output/results_nT{nT}_L{m}_MNone_rate{rate}_iseed{iseed}_new_heuristics.csv'
##                else:
##                    result_file = f'gurobi_energy_mathopt/output/results_nT{nT}_L{m}_MNone_rate{rate}_iseed{iseed}_new.csv'
#                df = pd.read_csv(result_file)
#                row = df.loc[df["hour"] == it] # bug fix
#                Cmin, Cmax, frob_norm = row["obj_val_min"].iloc[0], row["obj_val_max"].iloc[0], row["frobenius_norm"].iloc[0]
#            else:
#                # output new consumer list
#                sigma, Pt_ex, proc = gurobi_utils.load_covariance_and_expectation(USE_NEW, consumer_list, it, nT, rate)
#
#                # gurobi optimization
#                model = gurobi_utils.electric_power_demand_portfolio(m, sigma, Pt_ex, proc, None, nT, False, sense=gp.GRB.MINIMIZE)
#                model.optimize()
#                model_max = gurobi_utils.electric_power_demand_portfolio(m, sigma, Pt_ex, proc, None, nT, False, sense=gp.GRB.MAXIMIZE)
#                model_max.optimize()
#                Cmin, Cmax = model.ObjVal, model_max.ObjVal
#
#                # output interaction parameters
#                frob_norm = gurobi_utils.compute_qubo_frobenius_norm(sigma, Pt_ex, proc, nT)
#                dJ, dhex, shift = qubo2ising_correct(sigma, Pt_ex, proc)
#                dJ, dhex = dJ/frob_norm, dhex/frob_norm
#        else:
#            Mc_dict = {0:4, 3:7, 6:6, 9:6, 12:7, 15:6, 18:3, 21:5}
#            Mc = Mc_dict[it]
#            procp = 1.5
#            _, dJ, dhex, shift, _, _, frob_norm = param_PCE.param(m, nT, procp, it)
#
#            Cmin, _, Cmax, _, _, _, _, _ = read_param.Cminmax(it, Mc, procp)
#            Cmin /= m**2
#            Cmax /= m**2
#        return dJ, dhex, Cmin, Cmax, frob_norm, shift, consumer_list
#    else:
#        raise ValueError("Unsupported graph type")

