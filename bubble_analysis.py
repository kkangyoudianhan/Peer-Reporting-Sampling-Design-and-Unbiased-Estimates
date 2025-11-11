#!/usr/bin/env python3
import os
import re
import pickle
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

# 导入估计器
from estimators_v5 import (
    simplified_two_stage_sampling,
    weighted_ecm_estimate,
    ecmr_corrected_estimate,
    edge_balance_estimate,
    calculate_network_metrics
)

# -------- 全局配置 --------
SAMPLE_SIZE = 1200
NUM_TRIALS = 500
MAX_NEIGHBORS_OPTIONS = [5, 10]
COLOR_PALETTE = {'ECM': "#FF0000", 'ECMR': "#0000FF", 'ECM1': "#00CC00"}
AR_VALUES = [0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5]
H_VALUES = [-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25]
ESTIMATORS = ['ECM','ECMR','ECM1']

# -------- 单次实验 --------
def single_trial_worker(args):
    G, true_p, actual_ar, attr_dict, max_neighbors = args
    centers, obs = simplified_two_stage_sampling(
        G, sample_size=SAMPLE_SIZE, max_neighbors=max_neighbors, neighbor_sampling="partial"
    )
    if not centers or not obs["k_i"]:
        return []
    obs["attr_dict"] = attr_dict
    results = {
        "ECM": weighted_ecm_estimate(obs),
        "ECMR": ecmr_corrected_estimate(obs, R=actual_ar),
        "ECM1": edge_balance_estimate(obs, R=actual_ar, attr_dict=attr_dict),
    }
    rows = []
    for name, val in results.items():
        if pd.isna(val): continue
        err = val - true_p
        rows.append({
            "Estimator": name,
            "Error": err,
            "Abs_Error": abs(err),
            "max_neighbors": max_neighbors
        })
    return rows

# -------- compute 模式 --------
def run_compute_mode(network_file, output_dir, workers):
    filename = os.path.basename(network_file)
    match = re.match(r"H([+-]?\d+\.\d+)_AR(\d+\.\d+)\.pkl", filename)
    if not match:
        print(f"跳过: 文件名不符合格式 {filename}")
        return
    homo = float(match.group(1))
    nominal_ar = float(match.group(2))
    with open(network_file,"rb") as f:
        G = pickle.load(f)
    metrics = calculate_network_metrics(G)
    true_p = sum(1 for _,d in G.nodes(data=True) if d.get("attr")=="A")/G.number_of_nodes()
    actual_ar = metrics["activity_ratio"]
    attr_dict = {n:d.get("attr","B") for n,d in G.nodes(data=True)}

    all_rows=[]
    for max_n in MAX_NEIGHBORS_OPTIONS:
        tasks=[(G,true_p,actual_ar,attr_dict,max_n)]*NUM_TRIALS
        with multiprocessing.Pool(processes=workers) as pool:
            for res in tqdm(pool.imap_unordered(single_trial_worker,tasks),
                            total=NUM_TRIALS,desc=f"Trials max_n={max_n}"):
                for row in res:
                    row["H"]=homo
                    row["Nominal_AR"]=nominal_ar
                all_rows.extend(res)
    df=pd.DataFrame(all_rows)
    os.makedirs(output_dir,exist_ok=True)
    out=os.path.join(output_dir,filename.replace(".pkl","_results.pkl"))
    df.to_pickle(out)
    print(f"保存中间结果: {out}")

# -------- 绘图 --------
def create_bubble_plots(agg_df, outdir):
    for max_n in MAX_NEIGHBORS_OPTIONS:
        df=agg_df[agg_df["max_neighbors"]==max_n]
        if df.empty: continue
        max_err=df["Abs_Mean_Error"].max()
        scale=2000/max_err if max_err>0 else 2000
        # all-in-one
        plt.figure(figsize=(16,12))
        for est,color in COLOR_PALETTE.items():
            sub=df[df["Estimator"]==est]
            plt.scatter(sub["Nominal_AR"],sub["H"],s=sub["Abs_Mean_Error"]*scale,
                        c=color,alpha=0.6,label=est)
        plt.xticks(AR_VALUES); plt.yticks(H_VALUES)
        plt.xlim(0.4,2.6); plt.ylim(-0.35,0.3)
        plt.legend(); plt.grid(True,ls="--",alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,f"bubble_all_maxn{max_n}.png"),dpi=300)
        plt.close()

# -------- aggregate 模式 --------
def run_aggregate_mode(input_dir, output_dir):
    files=[f for f in os.listdir(input_dir) if f.endswith("_results.pkl")]
    dfs=[pd.read_pickle(os.path.join(input_dir,f)) for f in files]
    df=pd.concat(dfs,ignore_index=True)
    agg=(df.groupby(["H","Nominal_AR","max_neighbors","Estimator"],as_index=False)
          .agg(Mean_Error=("Error","mean"),Abs_Mean_Error=("Abs_Error","mean")))
    os.makedirs(output_dir,exist_ok=True)
    agg.to_csv(os.path.join(output_dir,"aggregated_results.csv"),index=False)
    create_bubble_plots(agg,output_dir)

# -------- 主入口 --------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["compute","aggregate"])
    ap.add_argument("--input",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--workers",type=int,default=1)
    args=ap.parse_args()
    if args.mode=="compute":
        run_compute_mode(args.input,args.output_dir,args.workers)
    else:
        run_aggregate_mode(args.input,args.output_dir)
