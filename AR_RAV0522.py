#F:\zhoumian\lower\V4\AR_RAV0420.py
import networkx as nx
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import re
from matplotlib.colors import LightSource

# 从 estimators_v5 导入指定函数
from estimators_v5 import (
    simplified_two_stage_sampling,
    weighted_ecm_estimate,
    edge_balance_estimate
)

# ================== 配置参数 ==================

SAMPLE_SIZE = 1000
NUM_TRIALS = 500
# 自动获取指定文件夹下的所有网络文件
network_folder = r"F:\zhoumian\lower\network\AR_enhanced0521"
network_files = [
    os.path.join(network_folder, f)
    for f in os.listdir(network_folder)
    if f.endswith('.pkl') and not f.startswith('fallback')
]
OUTPUT_DIR = r"F:\zhoumian\lower\results\ECM_V4\PA_AR\AR_enhanced0928"
DPI = 300
sampling_strategies = [
    ('full', 100),     # 完全采样：抽取所有邻居
    ('partial', 5),    # 部分采样：随机采样k=5
    ('partial', 10),   # 部分采样：随机采样k=10
    ('weighted', 10)   # 度加权采样：加权k=10
]


# 指定P_A和AR的取值范围
PA_RANGE = [0.1]
AR_RANGE = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]


# ================== 网络指标计算模块 ==================

def calculate_network_metrics(G: nx.Graph) -> dict:
    """Calculate true network metrics: P_A, AR"""
    metrics = {}
    
    # Proportion of A-class nodes
    num_A = sum(1 for n in G.nodes() if G.nodes[n]['attr'] == 'A')
    metrics['P_A'] = num_A / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    
    # Activity Ratio (AR)
    A_degrees = [G.degree(n) for n in G.nodes() if G.nodes[n]['attr'] == 'A']
    B_degrees = [G.degree(n) for n in G.nodes() if G.nodes[n]['attr'] == 'B']
    
    if len(B_degrees) == 0 or np.mean(B_degrees) == 0:
        metrics['AR'] = 1.0
    else:
        metrics['AR'] = np.mean(A_degrees) / np.mean(B_degrees) if A_degrees else 0.0
    
    return metrics

def parse_filename(filename):
    """解析网络文件名中的PA和AR值，兼容 AR0.7_PA30 格式"""
    try:
        ar_match = re.search(r'AR([0-9.]+)', filename)
        pa_match = re.search(r'PA([0-9]+)', filename)

        ar = float(ar_match.group(1)) if ar_match else None
        pa = float(pa_match.group(1)) / 100 if pa_match else None  # 转为比例

        return pa, ar
    except Exception as e:
        print(f"[parse_filename] 无法解析 {filename}: {e}")
        return None, None


def find_closest_value(value, value_range):
    """Find closest value in the given range"""
    return value_range[np.abs(value_range - value).argmin()]

# ================== 核心处理模块 ==================
# ================== 核心处理模块 ==================
def process_single_network(path, num_trials, sample_size, max_neighbors, neighbor_sampling):
    """对单个网络文件进行多次采样，计算 Sample/ECM/ECMR/ECMR_DYNAMIC 的 Bias、RMSE、Pbest 指标"""
    with open(path, 'rb') as f:
        G = pickle.load(f)

    metrics = calculate_network_metrics(G)
    true_ar = metrics['AR']
    true_p = metrics['P_A']

    filename = os.path.basename(path)
    target_pa, target_ar = parse_filename(filename)

    pa_parsed = find_closest_value(true_p, np.array(PA_RANGE))
    ar_parsed = find_closest_value(true_ar, np.array(AR_RANGE))

    results = {'Sample': [], 'ECM': [], 'ECM1': []}
    errors = {'Sample_Error': [], 'ECM_Error': [], 'ECM1_Error': []}

    for _ in range(num_trials):
        try:
            centers, obs = simplified_two_stage_sampling(
                G,
                sample_size=sample_size,
                max_neighbors=max_neighbors,
                neighbor_sampling=neighbor_sampling
            )

            if not centers or 'k_i' not in obs or 'k_iA' not in obs:
                continue

            # Sample 估计（原始比例）
            if 'y_i' in obs:
                sample_est = float(np.mean(obs['y_i']))
            else:
                sample_est = float(np.mean([1 if G.nodes[n]['attr'] == 'A' else 0 for n in centers]))

            results['Sample'].append(sample_est)
            errors['Sample_Error'].append(abs(sample_est - true_p))

            # ECM系列估计器
            # 创建节点属性字典，用于edge_balance_estimate
            attr_dict = {n: G.nodes[n]['attr'] for n in G.nodes()}
            
            ecm = weighted_ecm_estimate(obs)
            ecm1 = edge_balance_estimate(obs, true_ar, attr_dict)

            ecm = float(np.mean(ecm)) if isinstance(ecm, (list, np.ndarray)) else float(ecm)
            ecm1 = float(np.mean(ecm1)) if isinstance(ecm1, (list, np.ndarray)) else float(ecm1)

            results['ECM'].append(ecm)
            results['ECM1'].append(ecm1)

            errors['ECM_Error'].append(abs(ecm - true_p))
            errors['ECM1_Error'].append(abs(ecm1 - true_p))

        except Exception as e:
            print(f"[跳过 trial] 文件: {filename}, 错误: {e}")
            continue

    # 有效样本数量
    total_valid = len(results['Sample'])

    # ==== Bias（mean ± std）
    def summary(arr):
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    sample_mean, sample_std = summary(results['Sample'])
    ecm_mean, ecm_std = summary(results['ECM'])
    ecm1_mean, ecm1_std = summary(results['ECM1'])

    # ==== Abs Error
    sample_err_mean, sample_err_std = summary(errors['Sample_Error'])
    ecm_err_mean, ecm_err_std = summary(errors['ECM_Error'])
    ecm1_err_mean, ecm1_err_std = summary(errors['ECM1_Error'])

    # ==== RMSE
    def rmse(arr): return float(np.sqrt(np.nanmean([(e - true_p) ** 2 for e in arr])))
    sample_rmse = rmse(results['Sample'])
    ecm_rmse = rmse(results['ECM'])
    ecm1_rmse = rmse(results['ECM1'])

    # ==== Pbest（哪个估计器最接近真实值）
    best_counts = np.zeros(3)
    for e0, e1, e2 in zip(results['Sample'], results['ECM'], results['ECM1']):
        d0, d1, d2 = abs(e0 - true_p), abs(e1 - true_p), abs(e2 - true_p)
        best_counts[np.argmin([d0, d1, d2])] += 1

    pbest = [cnt / total_valid * 100 if total_valid > 0 else 0.0 for cnt in best_counts]

    return {
        'Filename': filename,
        'True_PA': true_p,
        'True_AR': true_ar,
        'PA_parsed': pa_parsed,
        'AR_parsed': ar_parsed,
        'max_neighbors': max_neighbors,
        'sampling_strategy': neighbor_sampling,

        # ==== Sample
        'Sample_mean': sample_mean,
        'Sample_std': sample_std,
        'Sample_Error_mean': sample_err_mean,
        'Sample_Error_std': sample_err_std,
        'Sample_RMSE': sample_rmse,
        'Pbest_Sample (%)': pbest[0],

        # ==== ECM
        'ECM_mean': ecm_mean,
        'ECM_std': ecm_std,
        'ECM_Error_mean': ecm_err_mean,
        'ECM_Error_std': ecm_err_std,
        'ECM_RMSE': ecm_rmse,
        'Pbest_ECM (%)': pbest[1],
        
        # ==== ECM1
        'ECM1_mean': ecm1_mean,
        'ECM1_std': ecm1_std,
        'ECM1_Error_mean': ecm1_err_mean,
        'ECM1_Error_std': ecm1_err_std,
        'ECM1_RMSE': ecm1_rmse,
        'Pbest_ECM1 (%)': pbest[2],

        # ==== Pairwise Improvement
        'ECM_vs_ECM1': ecm_err_mean - ecm1_err_mean,
    }


def main():
    all_results = []
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(network_files, desc="Processing selected network files"):
        for strategy, max_neighbors in sampling_strategies:
            try:
                stats = process_single_network(
                    filepath,
                    NUM_TRIALS,
                    SAMPLE_SIZE,
                    max_neighbors=max_neighbors,
                    neighbor_sampling=strategy
                )
                all_results.append(stats)
            except Exception as e:
                print(f"Error processing file {filepath} with strategy={strategy}, max_neighbors={max_neighbors}: {e}")
                continue

    # ─── 1. 原始统计结果 DataFrame
    df = pd.DataFrame(all_results)

    # ─── 2. 构建格式化列
    def combine(mean_col, std_col, digits=4):
        """把均值/标准差列合并成 mean(std) 字符串"""
        return df.apply(
            lambda r: f"{r[mean_col]:.{digits}g}({r[std_col]:.{digits}g})",
            axis=1
        )

    def combine_rmse(rmse_col, pbest_col, digits=4):
        """把 RMSE / Pbest 列合并成 RMSE(Pbest%) 字符串"""
        return df.apply(
            lambda r: f"{r[rmse_col]:.{digits}g}({r[pbest_col]:.1f}%)",
            axis=1
        )

    df['Sample_Error'] = combine('Sample_Error_mean', 'Sample_Error_std')
    df['ECM_Error'] = combine('ECM_Error_mean', 'ECM_Error_std')
    df['ECM1_Error'] = combine('ECM1_Error_mean', 'ECM1_Error_std')

    df['Sample_RMSE'] = combine_rmse('Sample_RMSE', 'Pbest_Sample (%)')
    df['ECM_RMSE'] = combine_rmse('ECM_RMSE', 'Pbest_ECM (%)')
    df['ECM1_RMSE'] = combine_rmse('ECM1_RMSE', 'Pbest_ECM1 (%)')

    # ─── 3. 筛选出需要输出的列（格式化摘要）
    summary_cols = [
        'Filename',
        'sampling_strategy',
        'max_neighbors',
        'PA_parsed',
        'AR_parsed',
        'Sample_Error',
        'ECM_Error',
        'ECM1_Error',
        'Sample_RMSE',
        'ECM_RMSE',
        'ECM1_RMSE'
    ]
    summary_df = df[summary_cols]

    # ─── 4. 保存两份 Excel 文件（完整版 & 摘要版）
    df.to_excel(os.path.join(OUTPUT_DIR, 'table_ECM_ECM1_full.xlsx'), index=False)
    summary_df.to_excel(os.path.join(OUTPUT_DIR, 'table_ECM_ECM1_summary.xlsx'), index=False)

    print(f"✅ 已保存完整结果：table_ECM_ECM1_full.xlsx")
    print(f"✅ 已保存格式化摘要：table_ECM_ECM1_summary.xlsx")


if __name__ == "__main__":
    main()