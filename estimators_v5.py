#V4\estimators_v3.py
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Tuple, Dict, List

# =========================
# 抽样模块 (ego–alter)
# =========================
def simplified_two_stage_sampling(
    G: nx.Graph,
    sample_size: int,
    max_neighbors: int = 5,
    neighbor_sampling: str = 'partial'
) -> Tuple[list, Dict]:
    """
    ego–alter 两阶段抽样:
    1. 随机抽取 ego 节点
    2. 从 ego 的邻居里抽 alter (全部 or 部分)

    返回:
      centers: 抽到的 ego 列表
      observations: 观测字典，包括每个 ego 的度、抽到的邻居数、A 邻居数等
    """
    nodes = list(G.nodes(data=True))
    attr_dict = {n: d.get("attr", "B") for n, d in nodes}
    degree_dict = dict(G.degree())

    centers = []
    obs = {
        "centers": [],
        "k_i": [],              # ego 的真实度
        "k_iA": [],             # 本次抽到的邻居里 A 的个数
        "m_i": [],              # 本次抽到的邻居数
        "sampled_neighbors": [],
        "neighbor_degrees": defaultdict(list),
        "attr_dict": attr_dict  # 保存节点属性，供 ECM1 使用
    }

    total_sampled = 0
    while total_sampled < sample_size:
        center = random.choice(nodes)[0]  # 随机 ego
        neighbors = list(G.neighbors(center))
        k_i = len(neighbors)

        if k_i == 0:  # 孤立点跳过
            continue

        # 抽邻居
        if neighbor_sampling == "partial" and k_i > max_neighbors:
            sampled_neighbors = random.sample(neighbors, max_neighbors)
        else:
            sampled_neighbors = neighbors

        m_i = len(sampled_neighbors)
        k_iA = sum(1 for nbr in sampled_neighbors if attr_dict[nbr] == "A")

        total_sampled += m_i

        # 记录
        obs["centers"].append(center)
        obs["k_i"].append(k_i)
        obs["k_iA"].append(k_iA)
        obs["m_i"].append(m_i)
        obs["sampled_neighbors"].append(sampled_neighbors)

        for nbr in sampled_neighbors:
            obs["neighbor_degrees"][attr_dict[nbr]].append(degree_dict[nbr])

        centers.append(center)

        if total_sampled >= sample_size:
            break

    return centers, obs

# =========================
# ECM 系列估计器
# =========================
def weighted_ecm_estimate(obs: Dict) -> float:
    """
    ECM: ego 等权平均
    \hat{x} = (1/S) sum_i (k_iA_sample / m_i)
    """
    total, valid = 0.0, 0
    for k_iA, m_i in zip(obs["k_iA"], obs["m_i"]):
        if m_i > 0:
            total += k_iA / m_i
            valid += 1
    return total / valid if valid > 0 else 0.0


def ecmr_corrected_estimate(obs: Dict, R: float) -> float:
    """
    ECMR: 活跃度校正
    \hat{p} = x_hat / (R + (1-R)x_hat)
    """
    x_hat = weighted_ecm_estimate(obs)
    denom = R + (1 - R) * x_hat
    return x_hat if denom <= 1e-12 else x_hat / denom

def ecm_estimate(obs: Dict) -> float:
    return edge_ecm_estimate(obs)


def dynamic_R_estimation(observations: Dict) -> float:
    """
    动态估计活跃系数R (基于已采集到的邻居度信息):
    R = avg_degree_A / avg_degree_B
    """
    deg_A = observations['neighbor_degrees']['A']
    deg_B = observations['neighbor_degrees']['B']
    
    if not deg_B:
        return 1.0  # 避免除以零
    
    mean_A = np.mean(deg_A) if deg_A else 0.0
    mean_B = np.mean(deg_B) if deg_B else 1.0
    if mean_B < 1e-12:
        return 1.0
    return mean_A / mean_B


def ecmr_dynamic_estimate(obs: Dict) -> float:
    R     = dynamic_R_estimation(obs)
    x_hat = weighted_ecm_estimate(obs)        
    denom = R + (1 - R) * x_hat
    return x_hat if denom <= 1e-12 else x_hat / denom

def ecmr_estimate(observations: Dict, R: float) -> float:
    """
    新增：ECMR估计(传入固定R)，但将 x_bar 改为 ecm_estimate(即 \sum k_i^A / \sum k_i),
    同时保留同质性校正逻辑:
    
    1) x_bar = ecm_estimate(observations) = sum(k_i^A)/sum(k_i)
    2) denominator = R + (1-R)* x_bar
    3) 若 denominator过小 则退化为 x_bar

    """
    x_bar = ecm_estimate(observations)
    denominator = R + (1 - R) * x_bar
    
    if denominator <= 1e-6:
        return x_bar
    
    return x_bar / denominator


# ----------------------------
# 优化的Bootstrap模块
# ----------------------------
def accelerated_bootstrap(
    observations: Dict,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    parallel: bool = False
) -> Tuple[float, Tuple]:
    """
    优化的Bootstrap置信区间计算
    改进点：
    - 向量化计算
    - 可选并行加速
    - 内存优化
    """
    k_i = np.array(observations['k_i'])
    k_iA = np.array(observations['k_iA'])
    # 此处示例：动态估计 R 以进行 ECMR 估计
    R = dynamic_R_estimation(observations)
    n = len(k_i)
    
    # 向量化Bootstrap
    boot_indices = np.random.choice(n, size=(n_bootstraps, n), replace=True)
    
    # 并行计算
    if parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(_single_bootstrap)(boot_idx, k_i, k_iA, R)
            for boot_idx in boot_indices
        )
    else:
        results = [_single_bootstrap(idx, k_i, k_iA, R) for idx in boot_indices]
    
    estimates = np.array(results)
    
    # 计算点估计 & 置信区间
    point_estimate = ecmr_corrected_estimate(observations, R)
    lower = np.percentile(estimates, 100 * alpha/2)
    upper = np.percentile(estimates, 100 * (1 - alpha/2))
    return point_estimate, (lower, upper)

def _single_bootstrap(indices, k_i, k_iA, R):
    """单次Bootstrap快速计算 (用于 ECMR)"""
    boot_k_i = k_i[indices]
    boot_k_iA = k_iA[indices]
    
    valid = (boot_k_i > 0)
    if valid.sum() == 0:
        return 0.0
    
    x_bar = (boot_k_iA[valid] / boot_k_i[valid]).mean()
    denominator = R + (1 - R) * x_bar
    if denominator < 1e-6:
        return x_bar
    return x_bar / denominator

# ----------------------------
# 网络分析工具 (可选)
# ----------------------------
def calculate_network_metrics(G: nx.Graph) -> Dict:
    """
    计算网络指标用于后续分析：
    - 网络密度
    - 平均聚集系数
    - 同质性 (Homophily)
    - 活跃系数 (R)
    """
    metrics = {}
    
    # 网络密度
    metrics['density'] = nx.density(G)

    # 平均聚集系数
    metrics['clustering'] = nx.average_clustering(G)
    
    # 计算同质性
    A_nodes = [n for n, attr in G.nodes(data='attr') if attr == 'A']
    E_AA = 0
    E_AB = 0
    
    for u, v in G.edges():
        u_attr = G.nodes[u].get('attr', 'B')
        v_attr = G.nodes[v].get('attr', 'B')
        if u_attr == 'A' and v_attr == 'A':
            E_AA += 1
        elif (u_attr == 'A' and v_attr == 'B') or (u_attr == 'B' and v_attr == 'A'):
            E_AB += 1
    
    # 同质性：A-A边占 (A-A + A-B) 的比例
    denominator = (E_AA + E_AB)
    if denominator > 0:
        metrics['homophily'] = E_AA / denominator
    else:
        metrics['homophily'] = 0.0
    
    # 活跃系数 R = A类节点平均度 / B类节点平均度
    deg_A = [G.degree(n) for n in A_nodes]
    deg_B = [G.degree(n) for n, attr in G.nodes(data='attr') if attr == 'B']
    
    if len(deg_B) == 0 or np.mean(deg_B) < 1e-12:
        metrics['activity_ratio'] = 1.0
    else:
        metrics['activity_ratio'] = (np.mean(deg_A) / np.mean(deg_B)) if deg_A else 0.0
    
    return metrics

def winsorize_observations(observations: Dict, threshold: float = 3.0) -> Dict:
    """
    Winsorize缩尾处理, 用于抑制极端度数(k_i)造成的异常值
    """
    k_i = np.array(observations['k_i'])
    k_iA = np.array(observations['k_iA'])
    
    if np.std(k_i) < 1e-12:
        return observations
    
    z_scores = (k_i - np.mean(k_i)) / np.std(k_i)
    outliers = np.abs(z_scores) > threshold
    
    upper = np.percentile(k_i, 95)
    lower = np.percentile(k_i, 5)
    k_i[outliers] = np.clip(k_i[outliers], lower, upper)
    
    # 保持 k_iA/k_i 的比例不变
    ratios = k_iA / observations['k_i']
    k_iA = np.round(k_i * ratios).astype(int)
    
    # 重组回原dict
    new_obs = {
        'k_i': k_i.tolist(),
        'k_iA': k_iA.tolist()
    }
    for k, v in observations.items():
        if k not in ['k_i', 'k_iA']:
            new_obs[k] = v
    return new_obs

def calculate_confidence_interval(
    observations: Dict,
    estimator_func,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    **kwargs
) -> Tuple[float, Tuple[float, float]]:
    """基于单次试验观测数据的Bootstrap CI计算"""
    # 首先计算原始点估计
    try:
        point_estimate = estimator_func(observations, **kwargs)
    except Exception as e:
        return np.nan, (np.nan, np.nan)
    
    estimates = []
    n = len(observations['k_i'])
    
    if n == 0:
        return np.nan, (np.nan, np.nan)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        boot_obs = {
            'k_i': [observations['k_i'][i] for i in indices],
            'k_iA': [observations['k_iA'][i] for i in indices],
            'centers': [observations['centers'][i] for i in indices],
            'neighbor_degrees': observations['neighbor_degrees']
        }
        try:
            est = estimator_func(boot_obs, **kwargs)
            estimates.append(est)
        except Exception as e:
            continue
    
    if len(estimates) < 2:
        return point_estimate, (point_estimate, point_estimate)
    
    # 使用原始点估计，不覆盖
    lower = np.nanpercentile(estimates, 100 * alpha/2)
    upper = np.nanpercentile(estimates, 100 * (1 - alpha/2))
    
    return point_estimate, (lower, upper)


def bootstrap_variance(observations, estimator_fn, B=500):
    """
    对单个观测数据进行 B 次 Bootstrap 重抽样，计算估计量的方差
    增强错误处理，确保即使某些样本失败也能得到有效结果
    """
    estimates = []
    n = len(observations.get('centers', []))
    
    # 如果样本太少，直接返回0
    if n <= 1:
        return 0.0
    
    for _ in range(B):
        try:
            # 从观测数据中进行有放回的抽样
            indices = np.random.choice(n, size=n, replace=True)
            
            # 生成新的观测数据样本，确保所有键都被复制
            boot_obs = {}
            for key in observations:
                if key == 'neighbor_degrees':
                    # 特殊处理嵌套字典
                    boot_obs['neighbor_degrees'] = {}
                    for attr in observations['neighbor_degrees']:
                        if indices.size > 0 and len(observations['neighbor_degrees'][attr]) > 0:
                            # 确保有足够的元素可供选择
                            valid_indices = [i for i in indices if i < len(observations['neighbor_degrees'][attr])]
                            if valid_indices:
                                boot_obs['neighbor_degrees'][attr] = [observations['neighbor_degrees'][attr][i] for i in valid_indices]
                            else:
                                boot_obs['neighbor_degrees'][attr] = []
                        else:
                            boot_obs['neighbor_degrees'][attr] = []
                elif isinstance(observations[key], list) and len(observations[key]) > 0:
                    # 确保列表类型键有足够的元素
                    valid_indices = [i for i in indices if i < len(observations[key])]
                    if valid_indices:
                        boot_obs[key] = [observations[key][i] for i in valid_indices]
                    else:
                        boot_obs[key] = []
                else:
                    # 其他类型直接复制
                    boot_obs[key] = observations[key]
            
            # 检查 bootstrap 样本是否有效
            if not boot_obs.get('centers') or not boot_obs.get('k_i'):
                continue
            
            # 计算估计值
            estimate = estimator_fn(boot_obs)
            if not np.isnan(estimate):
                estimates.append(estimate)
        except Exception as e:
            # 记录错误但继续尝试其他样本
            # print(f"Bootstrap 抽样出错: {e}")
            continue
    
    # 如果没有有效估计，返回0
    if not estimates:
        return 0.0
        
    # 计算方差
    return np.var(estimates, ddof=1)
def edge_ecm_estimate(obs: Dict) -> float:
    """
    边加权 ECM: 按边数权重
    \hat{x}_edge = sum_i k_iA_sample / sum_i m_i
    """
    s_m = float(sum(obs["m_i"]))
    if s_m == 0:
        return 0.0
    return float(sum(obs["k_iA"])) / s_m

import numpy as np
from typing import Dict, List, Any

def edge_balance_estimate(obs: Dict, R: float, attr_dict: Dict = None) -> float:
    """
    ECM1 (边平衡估计器):
      P_A = P_BA / (P_BA + R * P_AB)

    其中
      P_AB = A ego 出发的边指向 B 的比例
      P_BA = B ego 出发的边指向 A 的比例

    需要:
      - obs["centers"]: List[节点ID]
      - obs["sampled_neighbors"]: List[List[邻居ID]] 与 centers 对齐
      - attr_dict: 节点ID -> "A"/"B"，若未传入，则尝试从 obs["attr_dict"] 获取
      - R: 真实或外部给定的活跃比 (mean_k_A / mean_k_B)

    返回:
      P_A（估计的 A 比例）；若信息不足或无效，返回 np.nan
    """
    if attr_dict is None:
        attr_dict = obs.get("attr_dict", {})

    # 必要键检查
    if "centers" not in obs or "sampled_neighbors" not in obs:
        return np.nan
    centers: List[Any] = obs["centers"]
    sampled_neighbors: List[List[Any]] = obs["sampled_neighbors"]
    if not isinstance(sampled_neighbors, list) or len(centers) != len(sampled_neighbors):
        return np.nan

    # 计数边类型
    edges_AB = edges_AA = 0
    edges_BA = edges_BB = 0

    for center, nbrs in zip(centers, sampled_neighbors):
        c_attr = attr_dict.get(center, "B")
        for nbr in nbrs:
            n_attr = attr_dict.get(nbr, "B")
            if c_attr == "A":
                if n_attr == "B":
                    edges_AB += 1
                else:
                    edges_AA += 1
            else:  # c_attr == "B"
                if n_attr == "A":
                    edges_BA += 1
                else:
                    edges_BB += 1

    denom_A = edges_AB + edges_AA
    denom_B = edges_BA + edges_BB
    # 基本可识别性检查
    if denom_A == 0 or denom_B == 0 or not np.isfinite(R):
        return np.nan

    P_AB = edges_AB / denom_A
    P_BA = edges_BA / denom_B

    denom = P_BA + R * P_AB
    if denom <= 1e-12:
        return np.nan
    return P_BA / denom

# 如需别名
ecm1_estimate = edge_balance_estimate
def ecm2_estimate(obs: Dict) -> float:
    """
    ECM2 估计器:
      P_A = (r * (kB_bar / kA_bar)) / (r * (kB_bar / kA_bar) + 1)

    其中:
      r = sum_i k_i^A / sum_j k_j^B
      kA_bar = 样本中 A 类 ego 的平均度
      kB_bar = 样本中 B 类 ego 的平均度
    """
    attr_dict = obs.get("attr_dict", {})
    centers = obs.get("centers", [])
    k_i = obs.get("k_i", [])
    k_iA = obs.get("k_iA", [])

    if not centers or not k_i or not k_iA:
        return np.nan

    # A 邻居总和 & B 邻居总和
    total_A_neighbors = sum(k_iA)
    total_B_neighbors = sum(k - kA for k, kA in zip(k_i, k_iA))
    if total_B_neighbors <= 0:
        return np.nan
    r = total_A_neighbors / total_B_neighbors

    # 计算 A/B ego 的平均度
    kA_list = [k for ego, k in zip(centers, k_i) if attr_dict.get(ego, "B") == "A"]
    kB_list = [k for ego, k in zip(centers, k_i) if attr_dict.get(ego, "B") == "B"]

    if not kA_list or not kB_list:
        return np.nan

    kA_bar = np.mean(kA_list)
    kB_bar = np.mean(kB_list)
    if kA_bar <= 1e-12:
        return np.nan

    ratio = kB_bar / kA_bar
    denom = r * ratio + 1.0
    if denom <= 1e-12:
        return np.nan

    return (r * ratio) / denom
    
#pA_hat = ecm2_estimate(obs)