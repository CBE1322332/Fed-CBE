import numpy as np
import hdbscan
from copy import deepcopy
import torch
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import normclipping, prepare_updates, wrapup_aggregated_grads
from fl.models.model_utils import ol_from_vector
from aggregators import aggregator_registry
from sklearn.metrics.pairwise import cosine_distances


@aggregator_registry
class DeepSight(AggregatorBase):
    """
    [DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection](https://arxiv.org/abs/2201.00763) - NDSS '22
    DeepSight first calculates the Normalized Update Energy and Threshold Exceedings values for each client, and then calculates the cosine distance between the clients' bias updates. It then clusters the clients based on the cosine distance, NEUPs, and Division Differences, and clips the updates based on the median of the normed updates.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedAvg"
        """
        num_seeds (int): the number of DDifs for each client, aiming to reduce the randomness
        threshold_factor (float): the factor to determine the threshold for the NEUPs
        num_samples (int): the number of samples of noise dataset for DDif calculation
        tau (float): proportion threshold, 1/3. If the proportion of adversaries in a cluster is less than tau, the cluster is marked as benign.
        epsilon (float): a small value to avoid zero division
        """
        self.default_defense_params = {
            "num_seeds": 5,  # 增加种子数量以提高稳定性
            "threshold_factor": 0.005,  # 降低阈值因子使检测更严格
            "num_samples": 20000,
            "tau": 0.4,  # 提高 tau 值，防止过度拒绝
            "epsilon": 1.0e-6}
        self.update_and_set_attr()
        self.rand_datasets = self.generate_randdata()
        self.total_clients = args.total_clients  # 总客户端数量
        self.first_round = True  # 标记是否是第一轮

    def aggregate(self, updates, selected_client_ids=None, **kwargs):
        # 1. prepare model updates, gradient updates, output layers of gradient updates
        # load global model at last epoch
        self.global_epoch = kwargs.get('global_epoch', None)
        self.global_model = kwargs['last_global_model']
        
        # 获取当前轮次参与训练的客户端数量
        current_num_clients = len(updates)
        print(f"当前轮次参与训练的客户端数量: {current_num_clients}")
        if selected_client_ids is not None:
            print(f"本轮全局客户端ID: {selected_client_ids}")
        else:
            selected_client_ids = list(range(current_num_clients))
        
        # get model parameters updates and gradient updates
        client_updated_model, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False)
            
        # 准备输出层的更新
        if self.first_round:
            # 第一轮处理所有客户端
            self.ol_updates = np.array([
                ol_from_vector(
                    gradient_updates[cid], self.global_model, flatten=False, return_type='dict')
                for cid in range(self.total_clients)
            ])
            self.first_round = False  # 标记第一轮结束
        else:
            # 其他轮次只处理当前参与训练的客户端
            self.ol_updates = np.array([
                ol_from_vector(
                    gradient_updates[cid], self.global_model, flatten=False, return_type='dict')
                for cid in range(current_num_clients)
            ])

        # 2. filtering layer: prepare the NEUPs, DDifs, cosine distance for clustering
        NEUPs, TEs = self.get_NEUPs_TEs()
        DDifs = self.get_DDifs(client_updated_model)
        cosine_dists = self.get_cosine_distance()

        # 3. Ensemble clustering
        import warnings
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            cluster_labels = self.clustering(NEUPs, DDifs, cosine_dists)
        except RuntimeWarning as e:
            print(f"!Warning: {e}")

        # 4. Poisoned Cluster Identification
        suspicious_flags = TEs <= np.median(TEs)/3
        accepted_indices = []
        n_clusters = len(set(cluster_labels)) - \
            (1 if -1 in cluster_labels else 0)
        
        print(f"聚类数量: {n_clusters}")
        print(f"聚类标签: {cluster_labels}")
        
        for i in range(n_clusters):
            indices = np.where(cluster_labels == i)[0]
            amount_of_suspicious = np.sum(
                suspicious_flags[indices])
            # 输出全局ID
            print(f"聚类 {i} 中的全局客户端ID: {[selected_client_ids[idx] for idx in indices]}")
            print(f"聚类 {i} 中的可疑客户端数量: {amount_of_suspicious}")
            print(f"聚类 {i} 中的客户端总数: {len(indices)}")
            print(f"聚类 {i} 中的可疑客户端比例: {amount_of_suspicious/len(indices):.2f}")
            print(f"tau 阈值: {self.tau}")
            if amount_of_suspicious < self.tau * len(indices):
                accepted_indices.extend(indices)
                print(f"聚类 {i} 被接受")
            else:
                print(f"聚类 {i} 被拒绝")
        accepted_indices = np.array(accepted_indices, dtype=np.int64)

        num_anomalies = np.sum(suspicious_flags)
        print(f"异常更新数量: {num_anomalies}")

        anomaly_client_ids = np.where(suspicious_flags)[0]
        print(f"异常更新全局客户端ID: {[selected_client_ids[idx] for idx in anomaly_client_ids]}")
        print(f"可疑标志: {suspicious_flags}")
        print(f"TEs值: {TEs}")
        print(f"接受的全局客户端ID: {[selected_client_ids[idx] for idx in accepted_indices]}")

        # 5. clipping accepeted gradient updates w.r.t median of the L2 norm of all gradient updates
        clipped_gradient_updates = normclipping(gradient_updates[accepted_indices], np.median(
            np.linalg.norm(gradient_updates, axis=1)))

        # 6. aggregation
        return wrapup_aggregated_grads(clipped_gradient_updates, self.args.algorithm, self.global_model)

    def get_NEUPs_TEs(self):
        # get the Normalized Update Energy and Threshold Exceedings values for each client
        num_clients = self.total_clients if self.first_round else len(self.ol_updates)
        num_classes = self.args.num_classes
        TEs = np.empty(num_clients)
        NEUPs = np.empty((num_clients, num_classes))
        threshold_factor = max(self.threshold_factor,
                               1 / num_classes)

        # get the update of the weights and bias of the output layer
        for cid in range(num_clients):
            updates = self.ol_updates[cid]
            update_energy = np.abs(
                updates['bias']) + np.sum(np.abs(updates['weight']), axis=1)
            # NEUP, normalized update energy, actually the square of the L2 normalization to amplify differences
            energy_squared = update_energy**2
            NEUP = energy_squared/np.sum(energy_squared)
            # get Threshold Exceedings value
            threshold = threshold_factor * np.max(NEUP)
            TEs[cid] = np.sum(NEUP > threshold)
            NEUPs[cid] = NEUP

        return NEUPs, TEs

    def get_cosine_distance(self):
        # cosine distance
        num_clients = self.total_clients if self.first_round else len(self.ol_updates)
        bias_update = np.array([self.ol_updates[cid]['bias']
                                for cid in range(num_clients)])
        # 预处理：将非有限值替换为0
        bias_update = np.nan_to_num(bias_update, nan=0.0, posinf=0.0, neginf=0.0)
        
        cosine_dists = cosine_distances(
            bias_update.reshape(num_clients, -1))
        return cosine_dists.astype(np.float64)

    def get_DDifs(self, client_updated_model):
        # get the Division Differences
        num_clients = self.total_clients if self.first_round else len(self.ol_updates)
        DDifs = []
        for dataset in self.rand_datasets:
            seed_ddifs = []
            rand_loader = torch.utils.data.DataLoader(
                dataset, self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True
            )
            for cid in range(num_clients):
                client_updated_model[cid].eval()
                self.global_model.eval()

                DDif = torch.zeros(self.args.num_classes)
                # only random data as images/x for generating probability outputs
                for rand_images in rand_loader:
                    rand_images = rand_images.to(self.args.device)
                    with torch.no_grad():
                        output_client = client_updated_model[cid](rand_images)
                        output_global = self.global_model(rand_images)
                    # avoid zero-value
                    temp = output_client.cpu() / (output_global.cpu()+self.epsilon)
                    DDif.add_(torch.sum(temp, dim=0))
                seed_ddifs.append((DDif / self.num_samples).numpy())
            DDifs.append(seed_ddifs)
        return np.array(DDifs)

    def clustering(self, NEUPs, DDifs, cosine_dists):
        # classification
        def cluster_dists(statistic, precomputed=False):
            func = hdbscan.HDBSCAN(
                min_samples=3, metric='precomputed') if precomputed else hdbscan.HDBSCAN(min_samples=3)
            cluster_labels = func.fit_predict(statistic)
            cluster_dists = dists_from_clust(cluster_labels)
            return cluster_dists

        cosine_cluster_dists = cluster_dists(cosine_dists, precomputed=True)
        neup_cluster_dists = cluster_dists(NEUPs)
        ddif_cluster_dists = np.array(
            [cluster_dists(DDifs[i]) for i in range(self.num_seeds)])

        merged_ddif_cluster_dists = np.mean(ddif_cluster_dists, axis=0)
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)
        # allow_single_cluster=True, so that when there is no outlier, it will return a single cluster as 0, rather than -1 as outlier
        cluster_labels = hdbscan.HDBSCAN(
            metric='precomputed', allow_single_cluster=True, min_samples=3).fit_predict(merged_distances)
        return cluster_labels

    def generate_randdata(self):
        noise_shape = [self.args.num_channels,
                       self.args.num_dims, self.args.num_dims]
        # pre-generate random dataset for time saving
        rand_datasets = [
            RandDataset(noise_shape, self.num_samples, seed)
            for seed in range(self.num_seeds)
        ]
        return rand_datasets


class RandDataset(torch.utils.data.Dataset):
    def __init__(self, size, num_samples, seed):
        self.num_samples = num_samples
        torch.manual_seed(seed)  # Set the seed for reproducibility
        self.dataset = torch.rand(num_samples, *size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.dataset[idx]


def dists_from_clust(cluster_labels):
    # Create a boolean matrix where True indicates same cluster and False indicates different clusters
    same_cluster = (cluster_labels[:, None] == cluster_labels)
    # Convert to an integer matrix, where 1 represents different clusters, 0 represents the same cluster
    pairwise_dists = np.where(same_cluster, 0, 1)
    return pairwise_dists
