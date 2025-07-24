from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from fl.models.model_utils import model2vec, vec2model
from copy import deepcopy


@aggregator_registry
class TrustedAggregation(AggregatorBase):
    """
    TrustedAggregation uses a trusted model to compute trust scores for client updates and performs weighted aggregation based on these scores.
    """
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.eta = kwargs.get('eta', 1.0)  # 学习率，与原始实现保持一致
        self.trusted_model = None
        self.trusted_weights = None
        self.algorithm = "FedAvg"

    def set_trusted_model(self, trusted_model: nn.Module):
        """设置可信模型"""
        self.trusted_model = trusted_model
        self.trusted_weights = model2vec(trusted_model)

    def __center_model_(self, new: nn.Module, old: nn.Module):
        """将模型中心化，返回中心化后的模型参数"""
        centered_weights = []
        with torch.no_grad():
            for (_, new_weight), (_, old_weight) in zip(
                    new.state_dict().items(),
                    old.state_dict().items()
            ):
                # 计算中心化后的权重，但不直接修改模型
                centered_weight = new_weight - old_weight.detach().clone().to(new_weight.device)
                centered_weights.append(centered_weight)
        return centered_weights

    def __get_trust_score(self, trusted_model: nn.Module, user_model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算信任分数和缩放因子"""
        trusted_weights = []
        user_weights = []

        for (_, trusted_weight), (_, user_weight) in zip(
                trusted_model.state_dict().items(),
                user_model.state_dict().items()
        ):
            trusted_weights.append(torch.flatten(trusted_weight))
            user_weights.append(torch.flatten(user_weight))

        trusted_weights = torch.cat(trusted_weights)
        user_weights = torch.cat(user_weights)

        # 计算余弦相似度
        trusted_norm = torch.norm(trusted_weights)
        user_norm = torch.norm(user_weights)
        
        # 手动计算余弦相似度，确保数值稳定
        dot_product = torch.dot(trusted_weights, user_weights)
        norm_product = trusted_norm * user_norm
        cosine_sim = dot_product / (norm_product + 1e-8)  # 添加小量防止除零
        
        # 确保余弦相似度在[-1, 1]范围内
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        # 使用ReLU确保非负
        trust_score = F.relu(cosine_sim)
        
        # 计算范数比
        norm_ratio = trusted_norm / (user_norm + 1e-8)
        
        # 打印调试信息
        print(f"余弦相似度: {cosine_sim.item():.4f}, 范数比: {norm_ratio.item():.4f}")
        
        return trust_score, norm_ratio

    def aggregate(self, updates, **kwargs):
        """聚合客户端模型更新"""
        if self.trusted_model is None:
            raise ValueError("可信模型未设置")

        self.global_model = kwargs['last_global_model']
        # 获取原始设备
        device = next(self.global_model.parameters()).device
        
        # 确保模型在CPU上进行计算
        self.global_model = self.global_model.cpu()
        self.trusted_model = self.trusted_model.cpu()
        
        # 获取全局模型参数
        global_weights_vec = model2vec(self.global_model)
        
        # 保存原始模型参数
        original_global_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        # 获取客户端ID列表
        client_ids = kwargs.get('client_ids', [f'client_{i}' for i in range(len(updates))])
        
        client_models = []
        for update in updates:
            model = deepcopy(self.global_model)
            # 根据算法类型处理客户端上传的内容
            if self.algorithm == "FedSGD" or "FedOpt" in self.algorithm:
                # 如果是FedSGD或FedOpt，update是梯度更新
                vec2model(global_weights_vec + update, model)
            else:
                # 如果是FedAvg，update是完整模型参数
                vec2model(update, model)
            client_models.append(model)

        # 计算中心化后的模型参数
        trusted_centered = self.__center_model_(self.trusted_model, self.global_model)
        client_centered = [self.__center_model_(m, self.global_model) for m in client_models]

        print("\n=== 计算信任分数 ===")
        # 计算信任分数
        trust_obj = []
        for i, u_model in enumerate(client_models):
            print(f"\n计算客户端 {client_ids[i]} 的信任分数:")
            trust_score, trust_scale = self.__get_trust_score(self.trusted_model, u_model)
            trust_obj.append((trust_score, trust_scale))
        
        # 确保所有分数都是标量
        trust_scores = torch.tensor([t[0].item() for t in trust_obj])
        trust_scales = torch.tensor([t[1].item() for t in trust_obj])

        # 打印每个客户端的ID和权重
        print("\n=== 客户端聚合权重 ===")
        for client_id, score, scale in zip(client_ids, trust_scores, trust_scales):
            print(f"客户端 {client_id}: 信任分数 = {score.item():.4f}, 信任尺度 = {scale.item():.4f}")
        print("====================\n")

        # 更新全局模型
        with torch.no_grad():
            for (name, _), trusted_layer, *client_layers in zip(
                self.global_model.state_dict().items(),
                trusted_centered,
                *[c for c in client_centered]
            ):
                # 根据信任分数缩放权重
                trusted_obj = []
                for score, scale, layer in zip(trust_scores, trust_scales, client_layers):
                    # 确保所有操作数都是标量
                    scaled_layer = layer * score.item() * scale.item()
                    trusted_obj.append(scaled_layer)
                trusted_obj = torch.stack(trusted_obj)

                # 计算加权平均
                trusted_mean = trusted_obj.sum(dim=0)
                trust_sum = trust_scores.sum().item()  # 转换为标量
                trusted_mean = trusted_mean / trust_sum  # 标量除法

                # 更新参数
                self.global_model.state_dict()[name].copy_(
                    original_global_state[name] + self.eta * trusted_mean
                )

        # 将模型移回原始设备
        self.global_model = self.global_model.to(device)
        self.trusted_model = self.trusted_model.to(device)

        # 返回模型向量
        return model2vec(self.global_model)