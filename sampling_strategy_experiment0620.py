# F:\zhoumian\lower\V4\sampling_strategy_experiment.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import re
# 导入增强版估计器模块
from estimators_v4 import (
    enhanced_two_stage_sampling,
    weighted_ecm_estimate,
    ecmr_corrected_estimate,
    ecmr_dynamic_estimate,
    calculate_network_metrics
)

# ================================
# 配置参数
# ================================
# ================================
NETWORKS = [

    r"F:\zhoumian\ECMR\network_0419\soc-YouTube-ASU_network.pkl", # Added based on your notebook
]
# 采样策略
SAMPLING_STRATEGIES = [
    {'name': 'full', 'max_neighbors': None, 'weight_param': None},
    {'name': 'partial', 'max_neighbors': 10, 'weight_param': None},
    {'name': 'weighted', 'max_neighbors': 10, 'weight_param': 1.5}
]

# 估计器
ESTIMATORS = [
    {'name': 'ECM', 'func': weighted_ecm_estimate, 'kwargs': {}},
    {'name': 'ECMR_static', 'func': ecmr_corrected_estimate, 'kwargs': {'R': None}},
    {'name': 'ECMR_dynamic', 'func': ecmr_dynamic_estimate, 'kwargs': {}}
]

# 实验参数
NUM_TRIALS = 500

OUTPUT_DIR = r"F:\zhoumian\lower\results\ECM_V4\SamplingStrateg0507\0620"

# 图表配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
DPI = 300

# ================================
# 主要函数
# ================================
def parse_network_info(filepath):
    """增强版网络解析，支持多种文件命名格式"""
    filename = Path(filepath).stem
    
    # 用正则表达式分割下划线和连字符
    parts = re.split(r'[_-]', filename)
    network_type = parts[0]
    
    # 提取子类型（如果存在）
    if len(parts) > 1:
        subtype = '_'.join(parts[1:])
    else:
        subtype = None
    
    return {
        'network_type': f"{network_type}_{subtype}" if subtype else network_type,
        'name': filename,  # 添加完整文件名
        'n_nodes': 'N/A'   # 这个字段将在run_experiment中被真实值替换
    }

def run_experiment(network_file, sampling_strategy):
    """对单个网络运行特定采样策略的实验"""
    # 加载网络
    with open(network_file, 'rb') as f:
        G = pickle.load(f)
    
    # 计算真实网络指标
    metrics = calculate_network_metrics(G)
    true_p = metrics['P_A']
    true_AR = metrics['AR']
    n_nodes = metrics['n_nodes']  # 获取真实节点数
    sample_size = max(1, int(n_nodes * 0.1))  # 计算10%并确保至少1个样本
    # 解析网络信息
    network_info = parse_network_info(network_file)
    
    # 添加真实节点数和其他指标到network_info
    network_info['n_nodes'] = G.number_of_nodes()
    network_info['true_p_a'] = true_p  # 也可以直接在network_info中保存真实P(A)
    
    # 准备采样参数
    max_neighbors = sampling_strategy['max_neighbors']
    if sampling_strategy['name'] == 'full':
        max_neighbors = 10000  # 设置一个很大的值，确保获取所有邻居
    
    # 结果存储
    results = {est['name']: {'bias': [], 'estimates': []} for est in ESTIMATORS}
    
    # 多次重复实验
    for _ in tqdm(range(NUM_TRIALS), desc=f"真实P(A)={true_p:.2f}, 策略={sampling_strategy['name']}"):  # 关键修改处
        # 使用特定策略进行抽样
        centers, obs = enhanced_two_stage_sampling(
            G,
            sample_size=sample_size,
            max_neighbors=max_neighbors,
            neighbor_sampling=sampling_strategy['name'],
            weight_param=sampling_strategy.get('weight_param', 1.0)
        )
        
        # 如果抽样失败，跳过当前迭代
        if not centers:
            continue
        
        # 针对不同估计器计算结果
        for estimator in ESTIMATORS:
            # 复制kwargs以设置动态参数
            kwargs = estimator['kwargs'].copy()
            if 'R' in kwargs and kwargs['R'] is None:
                kwargs['R'] = true_AR
            
            # 计算估计值
            estimate = estimator['func'](obs, **kwargs)
            
            # 计算偏差
            bias = estimate - true_p
            
            # 记录结果
            results[estimator['name']]['estimates'].append(estimate)
            results[estimator['name']]['bias'].append(bias)
    
    # 计算汇总统计信息
    summary = {}
    for est_name, data in results.items():
        estimates = np.array(data['estimates'])
        biases = np.array(data['bias'])
        
        summary[est_name] = {
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
            'mean_bias': np.mean(biases),
            'std_bias': np.std(biases),
            'mean_abs_bias': np.mean(np.abs(biases)),
            'mse': np.mean(biases**2),
            'raw_estimates': estimates.tolist(),
            'raw_biases': biases.tolist()
        }
    
    return {
        'network_info': network_info,
        'true_values': metrics,
        'sampling_strategy': sampling_strategy['name'],
        'results': summary
    }

def save_detailed_results(all_results, output_dir):
    """保存详细实验结果到Excel文件"""
    rows = []
    
    for result in all_results:
        network_info = result['network_info']
        true_values = result['true_values']
        sampling_strategy = result['sampling_strategy']
        
        # 为每个估计器添加一行
        for est_name, stats in result['results'].items():
            row = {
                '网络类型': network_info['network_type'],
                '节点数': network_info['n_nodes'],
                'P(A)': true_values['P_A'], 
                '真实P(A)': true_values['P_A'],
                '真实AR': true_values['AR'],
                '采样策略': sampling_strategy,
                '估计器': est_name,
                '平均估计值': stats['mean_estimate'],
                '标准差': stats['std_estimate'],
                '平均偏差': stats['mean_bias'],
                '偏差标准差': stats['std_bias'],
                '平均绝对偏差': stats['mean_abs_bias'],
                'MSE': stats['mse']
            }
            rows.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_excel(os.path.join(output_dir, 'detailed_results.xlsx'), index=False)
    
    # 额外保存每种策略的汇总表
    for strategy in set(df['采样策略']):
        strategy_df = df[df['采样策略'] == strategy]
        summary_path = os.path.join(output_dir, f'summary_{strategy}.xlsx')
        strategy_df.to_excel(summary_path, index=False)
    
    return df
def save_all_raw_data_to_excel(all_results, output_dir, filename='all_raw_data.xlsx'):
    """
    将所有实验的原始估计值和偏差转换成一个 Excel 文件，
    不再区分采样策略或 P(A) 而分多个文件，而是全部写进一张表。
    
    使用：
        save_all_raw_data_to_excel(all_results, OUTPUT_DIR)
    
    保存后，你可以在 Excel 中按需筛选 [strategy]、[p_A] 等进行可视化。
    """
    rows = []
    
    for result in all_results:
        network_info = result['network_info']     # {'network_type':..., 'n_nodes':..., 'p_A':...}
        true_values = result['true_values']       # {'P_A':..., 'AR':...}
        sampling_strategy = result['sampling_strategy']
        
        # 获取一些字段供后续保存
        network_type = network_info['network_type']
        n_nodes = network_info['n_nodes']

        true_p = true_values['P_A']          # 真实 P(A)
        true_AR = true_values['AR']          # 真实 AR
        
        # 每个估计器都有若干次 trial 的数据
        for est_name, stats in result['results'].items():
            raw_estimates = stats['raw_estimates']
            raw_biases = stats['raw_biases']
            
            # trial 级别遍历
            for trial_idx, (est_val, bias_val) in enumerate(zip(raw_estimates, raw_biases)):
                row = {
                    'network_type': network_type,
                    'n_nodes': n_nodes,

                    'true_p': true_values['P_A'],  
                    'AR': true_AR,
                    'strategy': sampling_strategy,
                    'estimator': est_name,
                    'trial': trial_idx,      # 第几次实验
                    'estimate': est_val,     # 本次实验的估计值
                    'bias': bias_val
                }
                rows.append(row)
    
    # 转成 DataFrame
    df = pd.DataFrame(rows)
    # 可以排序一下，方便查看
    df.sort_values(by=['network_type','strategy','estimator','trial'], inplace=True)
    
    # 写出 Excel
    excel_path = os.path.join(output_dir, filename)
    df.to_excel(excel_path, index=False)
    
    print(f"[INFO] 已将所有原始数据保存到 {excel_path}")

# ================================
# 主程序
# ================================
def main():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 存储所有实验结果
    all_results = []
    
    # 对每个网络文件和每种采样策略运行实验
    for network_file in NETWORKS:
        for strategy in SAMPLING_STRATEGIES:
            result = run_experiment(network_file, strategy)
            all_results.append(result)
    

    
    # 保存原始数据
    save_all_raw_data_to_excel(all_results, OUTPUT_DIR, filename='all_raw_data.xlsx')
    



if __name__ == "__main__":
    main()