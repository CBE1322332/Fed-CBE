import torch
from attackers import attacker_registry
from attackers.pbases.dpbase import DPBase
from attackers.synthesizers.image_synthesizer import ImageSynthesizer
from fl.client import Client
from global_utils import actor
from .synthesizers import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning')
class BadNets(DPBase, Client):
    """
    BadNets is pixel pattern-based backdoor attack
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        # 先调用Client类的初始化方法，确保model属性被正确初始化
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # 然后初始化DPBase类的相关属性
        DPBase.__init__(self)
        
        """
        trigge_type: ["pattern_pixel", "single_pixel"]
        attack_model: ["targeted", "all2all", "all2one"]
        attack_strategy: ['single-shot', 'fixed-frequency','continuous']
        """
        # 从配置文件中读取攻击参数
        attack_params = None
        for attack in args.attacks:
            if attack['attack'] == 'BadNets':
                attack_params = attack['attack_params']
                break
                
        if attack_params is None:
            raise ValueError("BadNets attack parameters not found in config file")
            
        self.default_attack_params = {
            'trigger_size': attack_params['trigger_size'],
            "attack_model": attack_params['attack_model'],
            "poisoning_ratio": attack_params['poisoning_ratio'],
            "target_label": attack_params['target_label'],
            "source_label": 1,  # 保持默认值
            "attack_strategy": attack_params['attack_strategy'],
            "single_epoch": 0,  # 保持默认值
            "poison_frequency": 5,  # 保持默认值
            "attack_start_epoch": attack_params['attack_start_epoch']
        }
        self.update_and_set_attr()
        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency, self.attack_start_epoch)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

    def define_synthesizer(self):
        # initialize the backdoor synthesizer
        # single pixel trigger or pattern pixel trigger
        self.trigger = torch.ones((1, self.trigger_size, self.trigger_size))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)


@attacker_registry
@actor('attacker', 'data_poisoning')
class BadNets_image(DPBase, Client):
    """
    BadNets is pixel pattern-based backdoor attack
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        # 先调用Client类的初始化方法，确保model属性被正确初始化
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # 然后初始化DPBase类的相关属性
        DPBase.__init__(self)
        
        self.default_attack_params = {"trigger_path": "./attackers/triggers/trigger_white.png", "trigger_size": 5, "attack_model": "all2one",
                                      "poisoning_ratio": 0.32, "target_label": 6, "source_label": 1, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5}
        self.update_and_set_attr()
        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

    def define_synthesizer(self):
        self.synthesizer = ImageSynthesizer(
            self.args, self.trigger_path, self.trigger_size, self.attack_model, self.target_label, self.poisoning_ratio, self.source_label)
