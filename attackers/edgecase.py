import numpy as np
import torch
from attackers.pbases.dpbase import DPBase
from attackers.pbases.mpbase import MPBase
from datapreprocessor.edge_dataset import EdgeDataset
from global_utils import actor
from fl.models.model_utils import model2vec, vec2model
from attackers import attacker_registry
from .synthesizers import DatasetSynthesizer
from fl.client import Client

# TODO: test asr when pixel-based backdoor attack's bug is fixed


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class EdgeCase(MPBase, DPBase, Client):
    """
    [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) - NeurIPS '20
    Edge Case backdoor attack utilizes the edge-case samples viewed from the training dataset perspective (MNIST or CIFAR10) for edge-case backdoor embedding, and performs PGD and scaling attack to enhance the attack. Specifically, the attack steps include:
    1. edge-case data poisoning attack for MNIST and CIFAR10
        MNIST's edge dataset: label=7 images of ARDIS as label=7 of MNIST images; CIAFR10's edge dataset: southwest airline dataset as airplane of CIAFR10 Images
        self.args.target_label: the target label of the edge-case dataset
        for training: mix downsampled clean training dataset with edge-case dataset based on the poisoning_ratio
        for testing: edge-case dataset as full-poisoned test dataset and vanilla MNIST/CIFAR10 as vanilla test dataset
    2. L2 or L infinite norm-based PGD at the end of each local epoch
    2. model replacement attack, scaling attack
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        poisoning_ratio: ratio of edge data in the training dataset
        epsilon: Radius the l2 norm ball in PGD attack. For PGD with replacement, 0.25 for mnist, 0.083 for cifar10, coming from the paper
        projection_type: l_2 or l_inf
        l2_proj_frequency: projection frequency
        attack_start_epoch: 从哪一轮开始攻击，如果为None则从第0轮开始
        """
        self.default_attack_params = {
            "poisoning_ratio": 0.8, "epsilon": 0.25, "PGD_attack": True, "projection_type": "l_2", 
            "l2_proj_frequency": 1, "scaling_attack": True, "scaling_factor": 50, "target_label": 1,
            "attack_start_epoch": None
        }
        self.update_and_set_attr()

        # 保存原始训练数据集，用于后续创建加载器
        self.original_train_dataset = train_dataset
        
        self.define_synthesizer()
        
        # 检查当前轮次
        current_global_epoch = getattr(self.args, 'global_epoch', 0)
        attack_start_epoch = getattr(self, 'attack_start_epoch', 0)
        
        if current_global_epoch < attack_start_epoch:
            # 如果未达到攻击开始轮次，使用普通数据加载器
            self.logger.info(f"恶意客户端 {self.worker_id}：当前轮次 {current_global_epoch} 未达到攻击开始轮次 {attack_start_epoch}，使用普通数据加载器")
            self.train_loader = self.get_dataloader(train_dataset, train_flag=True, poison_epochs=False)
        else:
            # 如果已达到攻击开始轮次，使用带毒数据加载器
            self.logger.info(f"恶意客户端 {self.worker_id}：当前轮次 {current_global_epoch} 已达到攻击开始轮次 {attack_start_epoch}，使用带毒数据加载器")
            self.train_loader = self.get_dataloader(train_dataset, train_flag=True, poison_epochs=True)
            
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        self.synthesizer = DatasetSynthesizer(
            self.args, self.train_dataset, EdgeDataset(self.args, self.target_label), self.poisoning_ratio)
        # initialize the poisoned train dataset and test dataset
        self.poisoned_set = self.synthesizer.get_poisoned_set(
            train=True), self.synthesizer.get_poisoned_set(train=False)

    def get_dataloader(self, dataset, train_flag, poison_epochs=None, batch_size=None):
        # EdgeCase attack is this kind of attack using external prepared backdoor dataset
        poison_epochs = False if poison_epochs is None else poison_epochs
        data = self.poisoned_set[1 - train_flag] if poison_epochs else dataset
        # 如果未传入 batch_size，则使用 self.args.batch_size
        current_batch_size = batch_size if batch_size is not None else self.args.batch_size
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=current_batch_size, shuffle=train_flag, num_workers=self.args.num_workers, pin_memory=True)
        while True:  # train mode for infinite loop with training epoch as the outer
            for images, targets in dataloader:
                # 在测试模式下，确保 targets 是 torch.long 类型
                if not train_flag:
                    targets = targets.long()
                yield images, targets
            if not train_flag:
                # test mode for test dataset
                break
                
    def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
        # 检查当前轮次是否达到攻击开始轮次
        current_global_epoch = getattr(self.args, 'global_epoch', 0)
        attack_start_epoch = getattr(self, 'attack_start_epoch', 0)
        
        # 如果是首次达到攻击轮次，创建带毒数据加载器
        if current_global_epoch == attack_start_epoch:
            self.logger.info(f"恶意客户端 {self.worker_id}：首次达到攻击轮次，更新为带毒数据加载器")
            self.train_loader = self.get_dataloader(
                self.original_train_dataset, train_flag=True, poison_epochs=True)
        
        # 执行训练
        return super().local_training(model, train_loader, optimizer, criterion_fn, local_epochs)

    def step(self, optimizer, **kwargs):
        # PGD after step at each local epoch
        # normal step
        cur_local_epoch = kwargs["cur_local_epoch"]
        super().step(optimizer)
        
        # 检查当前轮次是否达到攻击开始轮次
        current_global_epoch = getattr(self.args, 'global_epoch', 0)
        attack_start_epoch = getattr(self, 'attack_start_epoch', 0)
        
        # 如果未达到攻击开始轮次，不执行PGD和模型修改
        if current_global_epoch < attack_start_epoch:
            return
            
        # 如果已达到攻击开始轮次，执行PGD和模型修改
        # get the updated model
        model_update = model2vec(self.model)
        w_diff = model_update - self.global_weights_vec

        # PGD projection
        if self.projection_type == "l_inf":
            smaller_idx = np.less(w_diff, -self.epsilon)
            larger_idx = np.greater(w_diff, self.epsilon)
            model_update[smaller_idx] = self.global_weights_vec[smaller_idx] - self.epsilon
            model_update[larger_idx] = self.global_weights_vec[larger_idx] + self.epsilon
        elif self.projection_type == "l_2":
            w_diff_norm = np.linalg.norm(w_diff)
            if (cur_local_epoch % self.l2_proj_frequency == 0 or cur_local_epoch == self.local_epochs - 1) and w_diff_norm > self.epsilon:
                model_update = self.global_weights_vec + self.epsilon * w_diff / w_diff_norm

        # load the model_update to the model after PGD projection
        vec2model(model_update, self.model)

    def non_omniscient(self):
        # scaling attack (model replacement attacks)
        # non_omniscient function is after the get_local_update function
        
        # 检查当前轮次是否达到攻击开始轮次
        current_global_epoch = getattr(self.args, 'global_epoch', 0)
        attack_start_epoch = getattr(self, 'attack_start_epoch', 0)
        
        # 如果未达到攻击开始轮次，不执行缩放攻击
        if current_global_epoch < attack_start_epoch:
            return self.update
            
        # 如果已达到攻击开始轮次，执行缩放攻击
        if self.scaling_attack:
            scaled_update = self.global_weights_vec + self.scaling_factor * \
                (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        else:
            scaled_update = self.update
        return scaled_update
