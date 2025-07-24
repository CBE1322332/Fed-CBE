from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry
import logging

logger = logging.getLogger(__name__)

@aggregator_registry
class RLR(AggregatorBase):
    """
    RLR (Robust Learning Rate) defense mechanism
    Uses geometric median to aggregate client updates
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedAvg"
        self.maxiter = 4  # 最大迭代次数
        self.eps = 1e-5  # 数值稳定性参数
        self.ftol = 1e-6  # 收敛阈值

    def weighted_average_oracle(self, points, weights):
        """计算加权平均"""
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """计算L2距离"""
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """计算几何中位数目标函数"""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        # 获取模型参数更新和梯度更新
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 计算每个客户端的数据点数量
        num_dps = np.array([len(update) for update in updates])
        alphas = num_dps / np.sum(num_dps)  # 权重

        # 使用加权平均作为初始中位数
        median = self.weighted_average_oracle(gradient_updates, alphas)
        num_oracle_calls = 1

        # 记录目标函数值
        obj_val = self.geometric_median_objective(median, gradient_updates, alphas)
        logger.info(f"Initial objective value: {obj_val}")

        # Weiszfeld算法迭代
        for i in range(self.maxiter):
            prev_median, prev_obj_val = median, obj_val
            
            # 计算权重
            weights = np.asarray([alpha / max(self.eps, self.l2dist(median, p)) 
                                for alpha, p in zip(alphas, gradient_updates)])
            weights = weights / weights.sum()
            
            # 更新中位数
            median = self.weighted_average_oracle(gradient_updates, weights)
            num_oracle_calls += 1
            
            # 计算新的目标函数值
            obj_val = self.geometric_median_objective(median, gradient_updates, alphas)
            
            # 检查收敛
            if abs(prev_obj_val - obj_val) < self.ftol * obj_val:
                logger.info(f"Converged after {i+1} iterations")
                break
            
            logger.info(f"Iteration {i+1}, Objective value: {obj_val}")

        logger.info(f"Final number of oracle calls: {num_oracle_calls}")
        logger.info(f"Final objective value: {obj_val}")

        # 将结果转换为float32类型
        agg_grad_updates = median.astype(np.float32)

        return wrapup_aggregated_grads(agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True) 