import torch
from attackers import attacker_registry
from attackers.pbases.dpbase import DPBase
from fl.client import Client
from global_utils import actor
from attackers.synthesizers import PixelSynthesizer

def get_trigger_pattern(args, adversarial_index):
    """
    获取触发器模式
    """
    # 从配置文件中获取trigger_num
    trigger_num = 4
    for attack in args.attacks:
        if attack['attack'] == 'FCBA':
            trigger_num = attack['attack_params'].get('trigger_num', 4)
            break
            
    if adversarial_index == -1:
        # 使用所有触发器
        patterns = []
        for i in range(trigger_num):
            patterns.extend(args.poison_patterns.get(f'{i}_poison_pattern', []))
        return patterns
    else:
        if adversarial_index < trigger_num:
            # 单个触发器
            return args.poison_patterns.get(f'{adversarial_index}_poison_pattern', [])
        elif adversarial_index < trigger_num * 2:
            # 相邻两个触发器的组合
            pattern1 = args.poison_patterns.get(f'{adversarial_index % trigger_num}_poison_pattern', [])
            pattern2 = args.poison_patterns.get(f'{(adversarial_index + 1) % trigger_num}_poison_pattern', [])
            return pattern1 + pattern2
        elif adversarial_index < trigger_num * 2 + 2:
            # 间隔两个触发器的组合
            pattern1 = args.poison_patterns.get(f'{adversarial_index % trigger_num}_poison_pattern', [])
            pattern2 = args.poison_patterns.get(f'{(adversarial_index + 2) % trigger_num}_poison_pattern', [])
            return pattern1 + pattern2
        else:
            # 三个触发器的组合
            pattern1 = args.poison_patterns.get(f'{(adversarial_index) % trigger_num}_poison_pattern', [])
            pattern2 = args.poison_patterns.get(f'{(adversarial_index + 1) % trigger_num}_poison_pattern', [])
            pattern3 = args.poison_patterns.get(f'{(adversarial_index + 2) % trigger_num}_poison_pattern', [])
            return pattern1 + pattern2 + pattern3

@attacker_registry
@actor('attacker', 'data_poisoning')
class FCBA(DPBase, Client):
    """
    FCBA: Full Combination Backdoor Attack
    """
    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        DPBase.__init__(self)
        attack_params = None
        for attack in args.attacks:
            if attack['attack'] == 'FCBA':
                attack_params = attack['attack_params']
                break
        if attack_params is None:
            raise ValueError("FCBA attack parameters not found in config file")
        self.default_attack_params = {
            'trigger_num': attack_params.get('trigger_num', 4),
            "attack_model": attack_params.get('attack_model', 'all2one'),
            "poisoning_ratio": attack_params.get('poisoning_ratio', 0.32),
            "target_label": attack_params.get('target_label', 7),
            "attack_strategy": attack_params.get('attack_strategy', 'continuous'),
            "single_epoch": attack_params.get('single_epoch', 0),
            "poison_frequency": attack_params.get('poison_frequency', 5),
            "attack_start_epoch": attack_params.get('attack_start_epoch', 10)
        }
        self.update_and_set_attr()
        
        # 设置触发器模式
        self.poison_patterns = {
            '0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
            '1_poison_pattern': [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
            '2_poison_pattern': [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
            '3_poison_pattern': [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
        }
        
        # 添加这一行，将poison_patterns添加到args中
        args.poison_patterns = self.poison_patterns
        
        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency, self.attack_start_epoch)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

    def define_synthesizer(self):
        # FCBA自定义触发器和synthesizer
        self.trigger = torch.ones((1, 5, 5))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model,
            target_label=self.target_label, poisoning_ratio=self.poisoning_ratio,
            source_label=None, single_epoch=self.single_epoch
        )

    def get_poison_batch(self, batch, adversarial_index=-1, evaluation=False):
        """
        获取中毒批次数据
        """
        # 获取触发器模式
        patterns = get_trigger_pattern(self.args, adversarial_index)
        
        # 创建新的张量，避免复制
        images, targets = batch
        batch_size = len(images)
        
        # 确定需要投毒的样本数量
        poison_size = batch_size if evaluation else int(batch_size * self.poisoning_ratio)
        
        # 批量处理投毒
        if poison_size > 0:
            # 修改标签
            targets[:poison_size] = self.target_label
            
            # 批量修改图像
            for pattern in patterns:
                images[:poison_size, :, pattern[0], pattern[1]] = 1
        
        # 移动到正确的设备
        images = images.to(self.args.device)
        targets = targets.to(self.args.device).long()
        
        return images, targets, poison_size
