import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datapreprocessor.cinic10 import CINIC10
from plot_utils import plot_label_distribution


def load_data(dataset):
    # Check if input is a string or an object with dataset attribute
    if isinstance(dataset, str):
        # Create a simple object to hold dataset name and other attributes
        class Args:
            pass
        args = Args()
        args.dataset = dataset
        
        # Set default values for CIFAR10
        if dataset == "CIFAR10":
            args.mean = [0.4914, 0.4822, 0.4465]
            args.std = [0.2470, 0.2435, 0.2616]
            args.model = "resnet18"
        # Add other dataset defaults as needed
        elif dataset == "MNIST":
            args.mean = [0.1307]
            args.std = [0.3081]
            args.model = "lenet"
        elif dataset == "FashionMNIST":
            args.mean = [0.2860]
            args.std = [0.3530]
            args.model = "lenet"
        else:
            # Set some reasonable defaults
            args.mean = [0.5, 0.5, 0.5]
            args.std = [0.5, 0.5, 0.5]
            args.model = "resnet18"
    else:
        # If it's an object, use it directly
        args = dataset
    
    # load dataset
    trans, test_trans = get_transform(args)
    data_directory = './data'
    if args.dataset == "EMNIST":
        train_dataset = datasets.EMNIST(data_directory, split="digits", train=True, download=True,
                                        transform=trans)
        test_dataset = datasets.EMNIST(
            data_directory, split="digits", train=False, transform=test_trans)
    elif args.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        train_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=True,
                                                        download=True, transform=trans)
        test_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=False,
                                                      download=True, transform=test_trans)
    elif args.dataset == "CINIC10":
        train_dataset = CINIC10(root=data_directory, train=True, download=True,
                               transform=trans)
        test_dataset = CINIC10(root=data_directory, train=False, download=True,
                              transform=test_trans)
    else:
        raise ValueError("Dataset not implemented yet")

    # deal with CIFAR10 list-type targets. CIFAR10 data is numpy array defaultly.
    train_dataset.targets = list_to_tensor(train_dataset.targets)
    test_dataset.targets = list_to_tensor(test_dataset.targets)
    
    # Return data info as well
    data_info = {
        "num_classes": 10,  # Default for CIFAR10, MNIST
        "mean": args.mean,
        "std": args.std,
        "num_channels": 3 if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10"] else 1
    }
    
    return train_dataset, test_dataset, data_info


def list_to_tensor(vector):
    """
    check whether a instance is tensor, convert it to tensor if it is a list.
    """
    if isinstance(vector, list):
        vector = torch.tensor(vector)
    return vector


def subset_by_idx(args, dataset, indices, train=True):
    trans = get_transform(args)[0] if train else get_transform(args)[1]
    dataset = Partition(
        dataset, indices, transform=trans)
    return dataset


def get_transform(args):
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST", "FEMNIST"] and args.model in ['lenet', "lr"]:
        # resize MNIST to 32x32 for LeNet5
        train_tran = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
        test_trans = train_tran
        # define the image dimensions for self.args, so that others can use it, such as DeepSight, lr model
        args.num_dims = 32
    elif args.dataset in ["CINIC10"]:
        train_tran = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)])
        test_trans = train_tran
    elif args.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        args.num_dims = 32 if args.dataset in ['CIFAR10', 'CIFAR100'] else 64
        
        # 检查是否启用数据增强
        enable_augmentation = getattr(args, 'enable_data_augmentation', True)  # 默认为True
        
        # 获取数据增强级别设置，默认为"basic"
        augmentation_level = getattr(args, 'augmentation_level', 'basic')
        
        if enable_augmentation:
            # 基础数据增强
            if augmentation_level == 'basic':
                train_tran = transforms.Compose([
                    transforms.RandomCrop(args.num_dims, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std)
                ])
            # 高级数据增强
            elif augmentation_level == 'advanced':
                train_tran = transforms.Compose([
                    transforms.RandomCrop(args.num_dims, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),  # 随机旋转±15度
                    transforms.ColorJitter(
                        brightness=0.2,  # 亮度变化范围
                        contrast=0.2,    # 对比度变化范围
                        saturation=0.2,  # 饱和度变化范围
                        hue=0.1          # 色调变化范围
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                    transforms.RandomErasing(p=0.2)  # 20%概率随机擦除部分区域
                ])
            # 最强数据增强
            elif augmentation_level == 'strong':
                train_tran = transforms.Compose([
                    transforms.RandomCrop(args.num_dims, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),  # 随机旋转±30度
                    transforms.ColorJitter(
                        brightness=0.3,  # 亮度变化范围更大
                        contrast=0.3,    # 对比度变化范围更大
                        saturation=0.3,  # 饱和度变化范围更大
                        hue=0.15         # 色调变化范围更大
                    ),
                    transforms.RandomGrayscale(p=0.1),  # 10%概率转为灰度图
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                    transforms.RandomErasing(p=0.3)  # 30%概率随机擦除
                ])
            else:
                # 默认情况，使用基础数据增强
                train_tran = transforms.Compose([
                    transforms.RandomCrop(args.num_dims, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std)
                ])
        else:
            # 不使用数据增强
            train_tran = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ])
            
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
    else:
        raise ValueError("Dataset not implemented yet")

    return train_tran, test_trans


def split_dataset(train_dataset, num_clients, distribution='iid', dirichlet_alpha=1.0):
    """
    Split dataset among clients
    
    Args:
        train_dataset: Training dataset to split
        num_clients: Number of clients
        distribution: Type of distribution ('iid' or 'non-iid')
        dirichlet_alpha: Alpha parameter for Dirichlet distribution in non-iid setting
    
    Returns:
        client_data: List of datasets for each client
    """
    print(f"Splitting dataset using {distribution} distribution...")
    
    if distribution == 'iid':
        # IID partition - divide data equally
        num_samples = len(train_dataset)
        samples_per_client = num_samples // num_clients
        client_data = []
        
        all_indices = torch.randperm(num_samples).tolist()
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
            indices = all_indices[start_idx:end_idx]
            
            client_dataset = Partition(train_dataset, indices)
            client_data.append(client_dataset)
    
    elif distribution == 'non-iid':
        # Non-IID partition using Dirichlet distribution
        indices_per_client = dirichlet_split_noniid(
            train_dataset.targets, dirichlet_alpha, num_clients)
        
        client_data = []
        for indices in indices_per_client:
            client_dataset = Partition(train_dataset, indices)
            client_data.append(client_dataset)
    
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    
    return client_data


def save_partition_cache(client_indices, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(client_indices, f)


def check_partition_cache(args):
    cache_exist = None
    folder_path = 'running_caches'
    file_name = f'{args.dataset}_balanced_{args.distribution}_{args.num_clients}_indices'
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        cache_exist = True if os.path.exists(file_path) else False
    return cache_exist, file_path


def check_noniid_labels(args, train_dataset, client_indices):
    """
    check the unique labels of each client and the common labels across all clients
    """
    client_unique_labels = {}
    common_labels = None
    for client_id, indices in enumerate(client_indices):
        # get the labels of the corresponding indices
        labels = train_dataset.targets[indices]
        # get the unique labels of the client
        unique_labels = set(labels.tolist())
        client_unique_labels[client_id] = unique_labels
        # for the first client, initialize common_labels as the unique labels
        if common_labels is None:
            common_labels = unique_labels
        else:
            # update common_labels by taking the intersection of the unique labels
            common_labels = common_labels.intersection(unique_labels)

    # log the unique labels of each client and the common labels across all clients
    args.logger.info(
        f"Common unique labels across all clients: {common_labels}")
    for client_id, unique_labels in client_unique_labels.items():
        args.logger.info(
            f"Client {client_id} has unique labels: {unique_labels}")


class Partition(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.classes = dataset.classes
        self.indices = indices if indices is not None else range(len(dataset))
        self.data, self.targets = dataset.data[self.indices], dataset.targets[self.indices]
        # (N, C, H, W) or (N, H, W) for MNIST-like grey images, mode='L'; CIFAR10-like color images, mode='RGB'
        self.mode = 'L' if len(self.data.shape) == 3 else 'RGB'
        self.transform = transform if transform is not None else dataset.transform
        self.poison = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # convert image to numpy array. for MNIST-like dataset, image is torch tensor, for CIFAR10-like dataset, image type is numpy array.
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy()
        # to return a PIL Image
        image = Image.fromarray(image, mode=self.mode)
        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有transform，则应用基本转换将PIL图像转为张量
            image = transforms.ToTensor()(image)

        # 确保target是张量
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        if self.poison:
            image, target = self.synthesizer.backdoor_batch(
                image, target.reshape(-1, 1))
            
        return image, target.squeeze()

    def poison_setup(self, synthesizer):
        self.poison = True
        self.synthesizer = synthesizer


def iid_partition(args, train_dataset):
    """
    nearly-quantity-balanced and class-balanced IID partition for clients.
    """
    labels = train_dataset.targets
    client_indices = [[] for _ in range(args.num_clients)]
    for cls in range(len(train_dataset.classes)):
        # get the indices of current class
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # get the number of sample class=cls indices for each client
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # random permutation
        class_indices = class_indices[torch.randperm(len(class_indices))]

        # calculate the number of samples for each client
        num_samples = len(class_indices)
        num_samples_per_client_per_class = num_samples // args.num_clients
        # other remaining samples
        remainder_samples = num_samples % args.num_clients

        # uniformly distribute the samples to each client
        for client_id in range(args.num_clients):
            start_idx = client_id * num_samples_per_client_per_class
            end_idx = start_idx + num_samples_per_client_per_class
            client_indices[client_id].extend(
                class_indices[start_idx:end_idx].tolist())
        # distribute the remaining samples to the first few clients
        for i in range(remainder_samples):
            client_indices[i].append(
                class_indices[-(i + 1)].item())
    client_indices = [torch.tensor(indices) for indices in client_indices]
    return client_indices


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Function: divide the sample index set into n_clients subsets according to the Dirichlet distribution with parameter alpha
    References:
    [orion-orion/FedAO: A toolbox for federated learning](https://github.com/orion-orion/FedAO)
    [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)
    '''
    # 确保train_labels是NumPy数组格式
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    elif not isinstance(train_labels, np.ndarray):
        train_labels = np.array(train_labels)
        
    n_classes = int(np.max(train_labels) + 1)
    # (K, N) category label distribution matrix X, recording the proportion of each category assigned to each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) records the sample index set corresponding to K classes
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # Record the sample index sets corresponding to N clients
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split divides the sample index k_idcs of class k into N subsets according to the proportion fracs
        # i represents the i-th client, idcs represents its corresponding sample index set
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def dataset_class_indices(dataset, class_label=None):
    num_classes = len(dataset.classes)
    if class_label:
        return torch.tensor(np.where(dataset.targets == class_label)[0])
    else:
        class_indices = [torch.tensor(np.where(dataset.targets == i)[
            0]) for i in range(num_classes)]
        return class_indices


def client_class_indices(client_indice, train_dataset):
    """
    Given the a client indice, return the list of indices of each class
    """
    labels = train_dataset.targets
    return [client_indice[labels[client_indice] == cls] for cls in range(len(train_dataset.classes))]


def class_imbalanced_partition(class_indices, im_iid_gamma, method='exponential'):
    """
    Perform exponential sampling on the number of each classes.

    Args:
        class_indices (list): A list of tensor containing index of each class for each client
        gamma (float): The exponential decay rate (0 < gamma <= 1).
        method (str, optional): The sampling method, exponential or step. Default as 'exponential'.

    Returns:
        sampled_class_indices (1d tensor): exponential-sampled class_indices
    """
    num_classes = len(class_indices)
    num_sample_per_class = [max(1, int(im_iid_gamma**(i / (num_classes-1)) * len(class_indices[i])))
                            for i in range(num_classes)]
    sampled_class_indices = [class_indices[i][torch.randperm(
        len(class_indices[i]))[:num_sample_per_class[i]]] for i in range(num_classes)]
    # print(f"num_sample_per_class: {num_sample_per_class}")
    return torch.cat(sampled_class_indices)


def create_client_test_subsets(client_data_list, test_dataset, test_subset_size=100):
    """
    为每个客户端创建专属的测试子集，基于其训练数据分布
    
    Args:
        client_data_list: 客户端训练数据列表，每个元素是一个客户端的Partition对象
        test_dataset: 全局测试集
        test_subset_size: 每个客户端测试子集的目标大小
        
    Returns:
        client_test_subsets: 每个客户端的测试子集字典，键为客户端ID，值为Subset对象
    """
    client_test_subsets = {}
    num_clients = len(client_data_list)
    test_labels = test_dataset.targets
    
    # 确保test_labels是torch.Tensor
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels)
    
    for client_id in range(num_clients):
        # 分析客户端训练数据的类别分布
        client_targets = client_data_list[client_id].targets
        # 确保client_targets是torch.Tensor
        if not isinstance(client_targets, torch.Tensor):
            client_targets = torch.tensor(client_targets)
        client_classes, client_class_counts = torch.unique(client_targets, return_counts=True)
        
        # 计算每个类别的比例
        client_class_distribution = client_class_counts.float() / torch.sum(client_class_counts.float())
        
        # 初始化这个客户端的测试子集索引
        client_test_indices = []
        
        # 按照客户端的类别分布从测试集中抽样
        for class_idx, class_ratio in zip(client_classes, client_class_distribution):
            # 计算应该为这个类别选择多少样本
            class_samples_count = max(1, int(test_subset_size * class_ratio))
            
            # 找到测试集中属于这个类别的所有样本索引
            class_indices = (test_labels == class_idx).nonzero(as_tuple=True)[0]
            
            # 如果没有足够的样本，使用所有可用样本
            if len(class_indices) <= class_samples_count:
                selected_indices = class_indices
            else:
                # 随机选择指定数量的样本
                perm = torch.randperm(len(class_indices))
                selected_indices = class_indices[perm[:class_samples_count]]
            
            # 添加到这个客户端的测试子集索引列表中
            client_test_indices.extend(selected_indices.tolist())
        
        # 确保测试子集大小不超过目标大小
        if len(client_test_indices) > test_subset_size:
            # 随机选择目标大小的样本
            perm = torch.randperm(len(client_test_indices))
            client_test_indices = [client_test_indices[i] for i in perm[:test_subset_size].tolist()]
        
        # 创建测试子集
        client_test_subsets[client_id] = torch.utils.data.Subset(test_dataset, client_test_indices)
        
    return client_test_subsets


if __name__ == "__main__":
    pass
