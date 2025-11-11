# F:\zhoumian\lower\V4\ECM_density_V4.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
import re

# 导入估计器
from estimators_v5 import (
    simplified_two_stage_sampling,
    weighted_ecm_estimate,
    ecmr_corrected_estimate,
    edge_balance_estimate
)

# ================================
# 配置参数
# ================================
NUM_TRIALS = 500
SAMPLE_SIZE = 1000
DPI = 300

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 结果数据结构
# ================================
results = {}

# ================================
# 文件名解析
# ================================
def parse_network_metadata(file_path):
    """解析文件名，提取网络类型、密度、真实R值"""
    filename = os.path.basename(file_path).replace('.pkl', '')
    # 支持格式: BA_d0.05_ar1.2.pkl
    match = re.search(r'(\w+)_d([\d\.]+)_ar([\d\.]+)', filename)
    if match:
        net_type = match.group(1)
        density = float(match.group(2))
        ar_value = float(match.group(3))
        return net_type, density, ar_value
    else:
        print(f"⚠️ 文件名解析失败: {filename}")
        return "Unknown", 0.0, 1.0

# ================================
# 核心处理函数
# ================================
def process_network(file_path):
    try:
        net_type, density, real_R = parse_network_metadata(file_path)

        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        nodes = list(G.nodes(data=True))
        true_p = len([n for n, attr in nodes if attr.get('attr') == 'A']) / len(nodes)

        # 初始化结果字典
        if net_type not in results:
            results[net_type] = {
                'ECM': defaultdict(list),
                'ECMR': defaultdict(list),
                'ECM1': defaultdict(list),
                'R_values': defaultdict(list)
            }

        # 多次实验
        for _ in tqdm(range(NUM_TRIALS), desc=f"处理 {os.path.basename(file_path)}"):
            centers, obs = simplified_two_stage_sampling(
                G,
                sample_size=SAMPLE_SIZE,
                max_neighbors=10,
                neighbor_sampling='partial'
            )

            if not centers or not obs['k_i']:
                continue

            # ECM
            ecm_val = weighted_ecm_estimate(obs)
            ecm_error = ecm_val - true_p

            # ECMR
            ecmr_val = ecmr_corrected_estimate(obs, R=real_R)
            ecmr_error = ecmr_val - true_p

            # ECM1
            attr_dict = {n: attr.get('attr', 'B') for n, attr in G.nodes(data=True)}
            ecm1_val = edge_balance_estimate(obs, R=real_R, attr_dict=attr_dict)
            ecm1_error = ecm1_val - true_p

            # 存储估计值和误差
            results[net_type]['ECM'][density].append((ecm_val, ecm_error))
            results[net_type]['ECMR'][density].append((ecmr_val, ecmr_error))
            results[net_type]['ECM1'][density].append((ecm1_val, ecm1_error))
            results[net_type]['R_values'][density].append(real_R)

    except Exception as e:
        print(f"\n处理 {file_path} 出错: {str(e)}")

# ================================
# 可视化
# ================================
def plot_density_comparison(output_dir, net_type):
    """绘制三种估计器误差的箱线图"""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)

    all_densities = sorted(results[net_type]['ECM'].keys())
    for est_name, color in zip(['ECM', 'ECMR', 'ECM1'], ['#FF0000', '#0000FF', '#00CC00']):
        # ✅ 修复这里：est -> est_name
        data = [[abs(err) for _, err in results[net_type][est_name][d]] for d in all_densities]
        positions = np.arange(1, len(all_densities) + 1) + (['ECM','ECMR','ECM1'].index(est_name) - 1) * 0.25
        ax.boxplot(data, patch_artist=True, positions=positions, widths=0.2,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   medianprops=dict(color='black'))
    
    ax.set_xticks(range(1, len(all_densities) + 1))
    ax.set_xticklabels([f"{d:.4f}" for d in all_densities], rotation=45)
    ax.set_xlabel('网络密度')
    ax.set_ylabel('绝对误差')
    ax.set_title(f'{net_type} 不同密度下估计误差对比')
    ax.legend(['ECM','ECMR','ECM1'])
    ax.grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(output_dir, f"{net_type}_Density_Comparison_Boxplot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📊 箱线图已保存至: {plot_path}")

# ================================
# 数据保存
# ================================
def save_enhanced_data(output_dir):
    """保存原始数据和统计量"""
    df_list = []
    for net_type in results:
        for density in results[net_type]['ECM']:
            for i in range(len(results[net_type]['ECM'][density])):
                record = {
                    'Network': net_type,
                    'Density': density,
                    'Trial': i + 1,
                    'ECM_Est': results[net_type]['ECM'][density][i][0],
                    'ECM_Error': results[net_type]['ECM'][density][i][1],
                    'ECMR_Est': results[net_type]['ECMR'][density][i][0],
                    'ECMR_Error': results[net_type]['ECMR'][density][i][1],
                    'ECM1_Est': results[net_type]['ECM1'][density][i][0],
                    'ECM1_Error': results[net_type]['ECM1'][density][i][1],
                    'R_value': results[net_type]['R_values'][density][i]
                }
                df_list.append(record)

    df = pd.DataFrame(df_list)
    os.makedirs(output_dir, exist_ok=True)

    # 原始数据
    raw_data_path = os.path.join(output_dir, "Enhanced_Results_RawData.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"✅ 原始数据已保存至: {raw_data_path}")

    # 统计数据
    stats = df.groupby(['Network', 'Density']).agg({
        'ECM_Error': ['mean', 'std'],
        'ECMR_Error': ['mean', 'std'],
        'ECM1_Error': ['mean', 'std'],
        'R_value': ['mean', 'std']
    })
    stats_path = os.path.join(output_dir, "Enhanced_Results_Statistics.csv")
    stats.to_csv(stats_path)
    print(f"✅ 统计数据已保存至: {stats_path}")

# ================================
# 主程序
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECM/ECMR/ECM1 在不同网络密度下的性能分析")
    parser.add_argument("--network_dir", type=str, required=True, help="包含.pkl网络文件的目录")
    parser.add_argument("--output_dir", type=str, required=True, help="保存结果的目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    network_files = [os.path.join(args.network_dir, f) for f in os.listdir(args.network_dir) if f.endswith('.pkl')]

    for path in tqdm(network_files, desc="总体进度"):
        process_network(path)

    save_enhanced_data(args.output_dir)
    for net_type in results:
        plot_density_comparison(args.output_dir, net_type)

    print(f"\n🎉 分析完成！结果保存在: {args.output_dir}")
