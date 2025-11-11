# -*- coding: utf-8 -*-
"""
Reality0930.py —— 现实网络抽样，支持 --net-id 参数。
），结果保存到 <OUTPUT_DIR>/<net_stem>/。

输出：
  <OUTPUT_DIR>/<net_stem>/
      <net_stem>_trial_records.csv
      <net_stem>_summary.xlsx
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import networkx as nx

# ========= 估计器与抽样 =========
from estimators_v5 import (
    simplified_two_stage_sampling,
    weighted_ecm_estimate,     # ECM
    edge_balance_estimate,     # ECM1
    ecm2_estimate              # ECM2
)
# ======= 顶部新增（或在 main() 前加）=======
import os
import matplotlib
matplotlib.use("Agg")  # 确保无显示环境可画图

# ======= 路径与参数 =======
# 自动扫描网络目录下的 .pkl
NETWORK_DIR = Path("/es01/paratera/sce1277/wenkang/ECM/network_0419")
NETWORK_PKLS = sorted([str(p) for p in NETWORK_DIR.glob("*.pkl")])

# 输出目录
OUTPUT_DIR = Path("/es01/paratera/sce1277/wenkang/ECM/reality")

# 试验次数（可被环境变量覆盖）
NUM_TRIALS = int(os.getenv("NUM_TRIALS", "1300"))

# 抽样策略同你原来的
STRATEGIES = {
    "F":   ("full",    10**9),
    "P5":  ("partial", 5),
    "P10": ("partial", 10),
    "W":   ("weighted", 10),
}

ESTIMATORS = ["ECM", "ECM1", "ECM2"]

SAMPLE_SIZES_FIXED = None
SAMPLE_SIZES_DYNAMIC_N = 12
SAMPLE_SIZE_MIN = 50
import networkx as nx
import pandas as pd

def load_network(pkl_path):
    try:
        G = nx.read_gpickle(pkl_path)   # 如果是 networkx 保存的 gpickle
        if not isinstance(G, nx.Graph):
            raise TypeError("Not a Graph")
        return G
    except Exception:
        # 如果不是 Graph，可能是 DataFrame，就直接跳过
        print(f"[Skip] {pkl_path} is not a valid Graph.")
        return None

# ========= 工具函数 =========
def calc_true_metrics(G: nx.Graph):
    """计算 True P(A), True AR"""
    for n in G.nodes():
        if 'attr' not in G.nodes[n]:
            G.nodes[n]['attr'] = 'B'
    A = [n for n in G.nodes() if G.nodes[n]['attr'] == 'A']
    B = [n for n in G.nodes() if G.nodes[n]['attr'] == 'B']
    p_true = len(A) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0
    degA = [G.degree(n) for n in A]
    degB = [G.degree(n) for n in B]
    mA = np.mean(degA) if degA else 0.0
    mB = np.mean(degB) if degB else 0.0
    if mB < 1e-12:
        R = 1.0 if mA < 1e-12 else 1e6
    else:
        R = mA / mB
    return p_true, R

def build_sample_sizes(n_nodes: int) -> list:
    """返回样本量列表"""
    if SAMPLE_SIZES_FIXED:
        return [s for s in SAMPLE_SIZES_FIXED if 0 < s <= n_nodes]
    upper = max(SAMPLE_SIZE_MIN, int(n_nodes * 0.10))
    upper = min(upper, n_nodes)
    xs = np.unique(np.linspace(SAMPLE_SIZE_MIN, upper, num=SAMPLE_SIZES_DYNAMIC_N, dtype=int)).tolist()
    if not xs:
        xs = [min(upper, n_nodes)]
    return xs

def one_estimator_value(est_name: str, obs: dict, R_true: float, attr_dict: dict) -> float:
    """统一计算 ECM / ECM1 / ECM2 点估计"""
    if est_name == "ECM":
        return float(weighted_ecm_estimate(obs))
    elif est_name == "ECM1":
        return float(edge_balance_estimate(obs, R=R_true, attr_dict=attr_dict))
    elif est_name == "ECM2":
        return float(ecm2_estimate(obs))
    else:
        return np.nan

def run_one_network(pkl_path: Path) -> dict:
    """对单个网络跑全部策略与样本量"""
    G = load_network(pkl_path)
    if G is None:
        print(f"跳过非图网络文件: {pkl_path}")
        return None
    if G.number_of_nodes() == 0:
        raise ValueError(f"空图：{pkl_path}")

    net_stem = Path(pkl_path).stem
    n_nodes = G.number_of_nodes()
    true_p, true_R = calc_true_metrics(G)
    attr_dict = {n: G.nodes[n].get('attr', 'B') for n in G.nodes()}
    sample_sizes = build_sample_sizes(n_nodes)

    trial_rows = []
    summary_rows = []

    for ss in tqdm(sample_sizes, desc=f"[{net_stem}] SampleSize", leave=False):
        seq_by_strategy = {code: {est: [] for est in ESTIMATORS} for code in STRATEGIES.keys()}
        for code, (neighbor_sampling, kmax) in STRATEGIES.items():
            for t in range(NUM_TRIALS):
                centers, obs = simplified_two_stage_sampling(
                    G,
                    sample_size=ss,
                    max_neighbors=kmax,
                    neighbor_sampling=neighbor_sampling
                )
                if not obs or not obs.get("centers"):
                    continue
                for est in ESTIMATORS:
                    val = one_estimator_value(est, obs, true_R, attr_dict)
                    if np.isnan(val):
                        continue
                    bias = abs(val - true_p)
                    trial_rows.append({
                        "network_type": net_stem,
                        "n_nodes": n_nodes,
                        "true_p": true_p,
                        "AR": true_R,
                        "strategy": neighbor_sampling,
                        "strategy_code": code,
                        "estimator": est,
                        "sample_size": ss,
                        "trial": t,
                        "estimate": val,
                        "bias": bias
                    })
                    seq_by_strategy[code][est].append(val)

        for code in STRATEGIES.keys():
            for est in ESTIMATORS:
                arr = np.asarray(seq_by_strategy[code][est], dtype=float)
                if arr.size == 0:
                    mean = std = min_v = max_v = bias_mean = np.nan
                else:
                    mean = float(np.mean(arr))
                    std  = float(np.std(arr))
                    min_v = float(np.min(arr))
                    max_v = float(np.max(arr))
                    bias_mean = float(np.mean(np.abs(arr - true_p)))
                summary_rows.append({
                    "network_type": net_stem,
                    "n_nodes": n_nodes,
                    "true_p": true_p,
                    "AR": true_R,
                    "sample_size": ss,
                    "strategy_code": code,
                    "estimator": est,
                    "mean": mean,
                    "std": std,
                    "min": min_v,
                    "max": max_v,
                    "bias_mean": bias_mean
                })

    trial_df = pd.DataFrame(trial_rows)
    summary_df = pd.DataFrame(summary_rows)
    return dict(
        net_stem=net_stem,
        n_nodes=n_nodes,
        true_p=true_p,
        true_R=true_R,
        trial_df=trial_df,
        summary_df=summary_df
    )
def plot_curve(pivot_df, true_p, net_name, output_dir_plots):
    if pivot_df.empty: return None
    plt.figure(figsize=(12, 8))
    x = pivot_df['SampleSize']
    for est in ESTIMATORS: # 这里会自动适应新的ESTIMATORS列表
        if f'{est}_mean' not in pivot_df.columns: continue
        plt.plot(x, pivot_df[f'{est}_mean'], color=COLORS.get(est, 'k'), marker='o', label=f'{est} 均值', lw=1.5)
        plt.fill_between(x, pivot_df[f'{est}_min'], pivot_df[f'{est}_max'], color=COLORS.get(est, 'grey'), alpha=ALPHAS.get(est, 0.15), label=f'{est} 波动范围')
    plt.axhline(true_p, color='grey', ls='--', lw=1.5, label=f'True P(A)={true_p:.3f}')
    plt.title(f'ECM / ECM1 / ECM2 估计性能 ({net_name})', fontsize=14) # 标题已更新
    plt.xlabel('Sample Size (Number of Neighbor Observations)')
    plt.ylabel('Estimate of P(A)')
    plt.grid(alpha=.3, ls='--'); plt.legend()
    plot_output_dir = Path(output_dir_plots) / "plots"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_output_dir / f'{net_name}.png'
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return out_png

def save_one_network_outputs(res: dict, out_dir: Path):
    """保存单网输出"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = res["net_stem"]

    trial_csv = out_dir / f"{stem}_trial_records.csv"
    res["trial_df"].to_csv(trial_csv, index=False)

    xlsx = out_dir / f"{stem}_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        if not res["summary_df"].empty:
            res["summary_df"].to_excel(w, sheet_name="统计摘要_summary", index=False)
        info_df = pd.DataFrame([{
            "Network": stem,
            "Nodes": res["n_nodes"],
            "True_P(A)": res["true_p"],
            "True_AR": res["true_R"],
            "Estimators": ",".join(ESTIMATORS),
            "Strategies": ",".join(STRATEGIES.keys())
        }])
        info_df.to_excel(w, sheet_name="网络信息", index=False)

# ======= main() 里改动：检查列表长度 =======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-id", type=int, required=True, help="网络索引 (0..N-1)")
    args = parser.parse_args()

    if len(NETWORK_PKLS) == 0:
        raise ValueError(f"NETWORK_DIR 为空：{NETWORK_DIR}")
    if not (0 <= args.net_id < len(NETWORK_PKLS)):
        raise ValueError(f"--net-id 必须在 0..{len(NETWORK_PKLS)-1} 之间")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = NETWORK_PKLS[args.net_id]
    print(f"\n=== Processing: {pkl} ===")

    res = run_one_network(pkl)
    if res is None:
        print(f"⚠️ 跳过: {pkl} 不是有效的网络图")
        return
    save_one_network_outputs(res, OUTPUT_DIR / res["net_stem"])
    print(f"✅ 完成: 结果已写入 {OUTPUT_DIR/res['net_stem']}")

if __name__ == "__main__":
    main()


#现在请你在这段代码里面增加绘图的函数（比如随着样本量变化，不同策略的估计值的均值和波动范围）