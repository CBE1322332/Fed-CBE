from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
import torch
import random


@aggregator_registry
class ClientDistillDefense(AggregatorBase):
    """客户端蒸馏防御方法

    该防御方法主要在客户端端进行，包括以下步骤：
    1. 使用随机标签数据进行遗忘训练
    2. 使用蒸馏数据集和知识蒸馏技术恢复主要任务能力
    3. 裁剪变化不明显的参数以进一步移除后门

    聚合端只进行简单的均值聚合
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            'distill_size': 1000,          # 蒸馏数据集大小
            'unlearn_epochs': 2,           # 遗忘训练轮数
            'distill_epochs': 5,           # 蒸馏训练轮数
            'ce_weight': 1.0,              # 交叉熵损失权重
            'kd_weight': 1.0,              # 知识蒸馏损失权重
            'attn_weight': 0.5,            # 注意力对齐损失权重
            'temperature': 4.0,            # 蒸馏温度（固定值）
            'prune_threshold': 0.01,       # 裁剪阈值，低于此阈值的参数变化将被裁剪为0
            'attn_layers': [],             # 需要进行注意力对齐的层名称，默认为空列表
            'random_label_ratio': 0.8,     # 随机标签比例
            'unlearn_lr_factor': 0.1,      # 遗忘训练学习率因子
            'unlearn_batch_limit': 5,      # 遗忘训练批次限制
            'distill_mode': 'logits',      # 蒸馏模式，可选'logits'或'features'
            'skip_normal_training': False, # 是否跳过正常本地训练
            'enable_pruning': True,        # 是否启用参数裁剪
            'defense_start_epoch': 15       # 开始防御的轮数
        }
        # 先使用标准方法从defense_params更新参数
        self.update_and_set_attr()
        # 然后检查并应用client_defense_params中的参数
        self.update_client_defense_params()
        
    def update_client_defense_params(self):
        """从args.client_defense_params更新参数，优先级高于args.defense_params"""
        if hasattr(self.args, 'client_defense_params') and self.args.client_defense_params:
            # 记录原始参数
            print(f"更新前的distill_size: {self.distill_size}")
            # 更新参数
            for key, value in self.args.client_defense_params.items():
                if key in self.default_defense_params:
                    setattr(self, key, value)
            # 记录更新后的参数
            print(f"更新后的distill_size: {self.distill_size}")

    def aggregate(self, updates, **kwargs):
        """简单的均值聚合，因为主要的防御在客户端进行"""
        return np.mean(updates, axis=0)
    
    def prepare_distill_dataset(self, server, train_dataset):
        """
        从训练集中准备蒸馏数据集和遗忘数据集
        
        参数:
            server: 服务器对象或包含model的字典
            train_dataset: 训练数据集
            
        返回:
            normal_distill_dataset: 正常标签(原始标签)的蒸馏数据集
            forgetting_dataset: 随机标签的遗忘数据集（使用与蒸馏数据集相同的样本）
        """
        # 保存当前随机状态
        python_state = random.getstate()
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        
        # 使用固定种子
        fixed_seed = 42
        random.seed(fixed_seed)
        torch.manual_seed(fixed_seed)
        np.random.seed(fixed_seed)
        
        try:
            # 获取训练集大小和类别数
            total_size = len(train_dataset)
            if hasattr(train_dataset, 'classes'):
                num_classes = len(train_dataset.classes)
            else:
                # 假设MNIST等数据集有10个类别
                num_classes = 10
                
            # 打印当前蒸馏数据集大小
            print(f"当前设置的蒸馏数据集大小: {self.distill_size}")
            
            # 确保蒸馏数据集大小不超过训练集大小
            distill_size = min(self.distill_size, total_size)
            print(f"实际使用的蒸馏数据集大小: {distill_size}")
            
            # 计算每个类别应选取的样本数
            samples_per_class = distill_size // num_classes
            
            # 使用对象属性而不是重新从server.args获取
            random_label_ratio = self.random_label_ratio

            # 按类别整理样本索引
            class_indices = [[] for _ in range(num_classes)]
            for idx in range(total_size):
                label = train_dataset.targets[idx].item() if hasattr(train_dataset, 'targets') else train_dataset[idx][1]
                class_indices[label].append(idx)
            
            # 从每个类别中均匀选择蒸馏数据集的样本
            selected_indices_distill = []
            for class_idx in range(num_classes):
                if len(class_indices[class_idx]) >= samples_per_class:
                    selected_indices_distill.extend(random.sample(class_indices[class_idx], samples_per_class))
                else:
                    # 如果某类别样本不足，全部选取并从其他类别补充
                    selected_indices_distill.extend(class_indices[class_idx])
            
            # 如果还需要更多样本才能达到目标大小，从所有类别中随机选择
            remaining = distill_size - len(selected_indices_distill)
            if remaining > 0:
                all_remaining_indices = []
                for class_idx in range(num_classes):
                    remaining_in_class = [idx for idx in class_indices[class_idx] if idx not in selected_indices_distill]
                    all_remaining_indices.extend(remaining_in_class)
                
                selected_indices_distill.extend(random.sample(all_remaining_indices, min(remaining, len(all_remaining_indices))))
            
            # 创建正常标签的蒸馏数据集（使用原始标签）
            normal_distill_dataset = torch.utils.data.Subset(train_dataset, selected_indices_distill)
            
            # 按类别整理选中的样本索引（用于创建遗忘数据集）
            selected_class_indices_distill = [[] for _ in range(num_classes)]
            for idx_pos, idx in enumerate(selected_indices_distill):
                label = train_dataset.targets[idx].item() if hasattr(train_dataset, 'targets') else train_dataset[idx][1]
                selected_class_indices_distill[label].append(idx_pos)
            
            # ----- 创建遗忘数据集（使用相同的样本但分配随机标签） -----
            
            # 使用与蒸馏数据集相同的样本
            forgetting_labels = []
            
            # 为每个类别分配随机标签
            for class_idx in range(num_classes):
                # 为该类别的所有样本分配随机标签（不同于原始标签）
                random_label = (class_idx + random.randint(1, num_classes-1)) % num_classes
                for idx_pos in selected_class_indices_distill[class_idx]:
                    forgetting_labels.append(random_label)
            
            # 创建随机标签的遗忘数据集
            forgetting_dataset = torch.utils.data.Subset(train_dataset, selected_indices_distill)
            forgetting_dataset.targets = torch.tensor(forgetting_labels)
            
            print(f"创建了蒸馏数据集 (原始标签): {len(normal_distill_dataset)} 个样本")
            print(f"创建了遗忘数据集 (随机标签): {len(forgetting_dataset)} 个样本")
            print(f"两个数据集使用相同的样本，但遗忘数据集分配了随机标签")
            
            # 为了向后兼容，返回None作为random_distill_dataset
            return normal_distill_dataset, None, forgetting_dataset
            
        finally:
            # 恢复随机状态
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)

    def get_attention(self, model, inputs, layer_names):
        """获取模型在指定层的注意力输出"""
        attentions = {}
        hooks = []
        
        # 定义 hook 函数
        def hook_fn(name):
            def hook(module, input, output):
                attentions[name] = output
            return hook
        
        # 注册 hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        # 前向传播
        _ = model(inputs)
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
            
        return attentions 

    def preprocess_model(self, model):
        """模型训练前的预处理（可选）
        
        本方法可以对模型进行预处理，例如初始化模型状态、添加正则化等
        """
        # 在这里实现模型预处理逻辑
        return model

 
    