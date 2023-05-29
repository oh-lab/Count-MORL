import numpy as np

class HashCount(object):
    """Hash-based count bonus for exploration.
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y., Schulman, J., De Turck, F., and Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in Neural Information Processing Systems (NIPS)
    """
    def __init__(self, dim_key=8, obs_dim=None, act_dim=None, bucket_sizes=None, projection_matrix=None):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]

        obs_mods_list = []
        for bucket_size in bucket_sizes:
            obs_mod = 1
            obs_mods = []
            for _ in range(dim_key):
                obs_mods.append(obs_mod)
                obs_mod = (obs_mod * 2) % bucket_size
            obs_mods_list.append(obs_mods)

        self.bucket_sizes = np.asarray(bucket_sizes)
        self.obs_mods_list = np.asarray(obs_mods_list).T
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))

        if projection_matrix is None:
            self.obs_projection_matrix = np.random.normal(size=(obs_dim, dim_key))
        else:
            self.obs_projection_matrix = projection_matrix
            
        self.obs_dim = obs_dim
        self.dict = {}

    def _compute_keys(self, obss):

        obs_round = np.round(np.asarray(obss, dtype=np.float128))
        keys = np.cast['int'](obs_round.dot(self.obs_mods_list)) % self.bucket_sizes

        keys_tuple = tuple(map(tuple, keys))
        if keys_tuple not in self.dict:
            self.dict[keys_tuple] = 1

        return keys

    def _inc_hash(self, obss):
        keys = self._compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def _query_hash(self, obss):
        keys = self._compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self._query_hash(obss)
        self._inc_hash(obss)

    def _predict(self, obs):
        counts = self._query_hash(obs)
        prediction = np.maximum(1.0, counts)
        return prediction

    def compute_intrinsic_reward(self, obs, train = True):
        if train:
            self.fit_before_process_samples(obs)
        
        if not train:
            reward = self._predict(obs)
            return np.expand_dims(reward, 1)

    def _sigmoid(self, obs):
        return 1/(1 + np.exp(-obs))

    def compute_cluster(self):
        return len(self.dict)