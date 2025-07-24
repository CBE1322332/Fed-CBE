from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry
import logging

logger = logging.getLogger(__name__)

@aggregator_registry
class Multi_metrics(AggregatorBase):
    """
    Multi_metrics defense mechanism
    Uses multiple distance metrics (cosine, Manhattan, and Euclidean) to detect and filter out malicious clients
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "p": 0.3,  # 默认选择30%的客户端
        }
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        
        # 获取本轮参与训练的客户端列表
        selected_clients = kwargs.get("selected_clients", None)
        if selected_clients is not None:
            # 从selected_clients中获取客户端ID
            current_round_clients = [client.worker_id for client in selected_clients]
        else:
            current_round_clients = list(range(len(updates)))
        
        # 直接使用模型参数，不进行梯度转换
        model_updates = updates

        # 计算三种距离度量
        cos_dis = [0.0] * len(model_updates)
        length_dis = [0.0] * len(model_updates)
        manhattan_dis = [0.0] * len(model_updates)

        for i, g_i in enumerate(model_updates):
            for j in range(len(model_updates)):
                if i != j:
                    g_j = model_updates[j]
                    
                    # 计算余弦距离
                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)
                    # 计算曼哈顿距离
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))
                    # 计算欧氏距离
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        # 组合三种距离度量
        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        # 计算协方差矩阵及其逆矩阵
        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        # 计算马氏距离
        ma_distances = []
        for i, g_i in enumerate(model_updates):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        logger.info(f"Client scores: {scores}")

        # 打印每个客户端的ID和分数
        print("\n=== 客户端分数详情 ===")
        for client_id, score in zip(current_round_clients, scores):
            print(f"客户端 {client_id}: 分数 = {score:.6f}")
        print("====================\n")

        # 选择得分最低的30%客户端
        p = 0.3
        p_num = p * len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]

        # 获取选中客户端的真实ID（从本轮参与训练的客户端中选择）
        selected_client_ids = [current_round_clients[i] for i in topk_ind]
        print(f"从本轮参与训练的客户端 {current_round_clients} 中选出的客户端ID: {selected_client_ids}")

        # 获取选中客户端的数据点数量
        selected_num_dps = np.array(kwargs.get("num_dps", [1] * len(model_updates)))[topk_ind]
        # 计算权重
        reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]
        
        logger.info(f"Selected clients data points: {selected_num_dps}")
        logger.info(f"Calculated weights: {reconstructed_freq}")

        # 使用数据量权重进行加权平均
        aggregated_update = np.average(np.array(model_updates)[topk_ind], weights=reconstructed_freq, axis=0)
        
        return aggregated_update 