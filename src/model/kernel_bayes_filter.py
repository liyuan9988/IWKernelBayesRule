import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
from torch import nn
from pathlib import Path
from sklearn.kernel_approximation import RBFSampler

from src.data.data_class import TimeSequence
from src.data import generate_sequence
from src.utils import cal_loocv_emb, cal_loocv
from src.utils.kernel_func import AbsKernel, GaussianKernel
from src.utils.pytorch_linear_reg_utils import linear_reg_loss
from src.feature_map import obtain_feature_model


class KBR:
    latent_kernel_mat: np.ndarray
    obs_kernel_mat: np.ndarray
    obs_test_kernel_mat: np.ndarray
    predict_weight: np.ndarray
    kernel_bayes_stage1_matrix: np.ndarray
    reg1: float
    reg2: float

    def __init__(self, train_length: int, kernel_bayes_rule: str, scale: float, reg2: float,
                 adaptive_feature: Optional[nn.Module] = None, lr: float = 0.01,
                 weight_decay: float = 0.01, **kwargs):
        self.train_length = train_length
        self.kernel_bayes_rule = kernel_bayes_rule
        self.scale = scale
        self.reg2 = reg2
        self.adaptive_feature = adaptive_feature
        self.lr = lr
        self.weight_decay = weight_decay
        assert self.kernel_bayes_rule in ("original", "proposal")

    def fit_and_test(self, entire_seq: TimeSequence):
        train_seq = TimeSequence(latent=entire_seq.latent[:self.train_length],
                                 obs=entire_seq.obs[:self.train_length])

        test_seq = TimeSequence(latent=entire_seq.latent[self.train_length:],
                                obs=entire_seq.obs[self.train_length:])

        latent_kernel_func = GaussianKernel()
        latent_kernel_func.fit(entire_seq.latent)
        latent_kernel_func.sigma *= self.scale
        self.latent_kernel_mat = latent_kernel_func.cal_kernel_mat(train_seq.latent, train_seq.latent)

        self.obs_kernel_mat, self.obs_test_kernel_mat = self.cal_obs_kernel_mat(train_seq, test_seq,
                                                                                latent_kernel_func.sigma)
        self.tune_stage1_reg()
        self.fit_predict_matrix()
        self.fit_kernel_bayes_stage1_matrix()

        weight = np.ones((self.train_length, 1)) / self.train_length

        predict = np.array([np.zeros(train_seq.latent.shape[1])])
        for i in range(test_seq.obs.shape[0]):
            # prediction
            weight = self.predict_weight @ weight
            # filtering
            gamma = self.kernel_bayes_stage1_matrix @ weight
            if self.kernel_bayes_rule == "proposal":
                weight = self.compute_filtering_proposed(gamma, self.obs_test_kernel_mat[:, [i]])
            else:
                weight = self.compute_filtering_original(gamma, self.obs_test_kernel_mat[:, [i]])
            pred_one = weight.T @ train_seq.latent
            predict = np.append(predict, pred_one, axis=0)

        return np.sum((predict[1:] - test_seq.latent) ** 2) / test_seq.latent.shape[0]

    def cal_obs_kernel_mat(self, train_seq, test_seq, latent_sigma) -> Tuple[np.ndarray, np.ndarray]:
        if self.adaptive_feature is None:
            obs_kernel_func = GaussianKernel()
            obs_kernel_func.fit(np.concatenate([train_seq.obs, test_seq.obs], axis=0))
            obs_kernel_func.sigma *= self.scale
            obs_kernel_mat = obs_kernel_func.cal_kernel_mat(train_seq.obs, train_seq.obs)
            obs_test_kernel_mat = obs_kernel_func.cal_kernel_mat(train_seq.obs, test_seq.obs)
            return obs_kernel_mat, obs_test_kernel_mat

        # Fit adaptive feature
        rbf_sampler = RBFSampler(gamma=1.0 / latent_sigma).fit(train_seq.latent)
        train_obs_data_t = torch.tensor(train_seq.obs, dtype=torch.float32)
        train_latent_feature = rbf_sampler.transform(train_seq.latent)
        train_latent_feature_t = torch.tensor(train_latent_feature, dtype=torch.float32)
        test_obs_data_t = torch.tensor(test_seq.obs, dtype=torch.float32)

        opt = torch.optim.Adam(self.adaptive_feature.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

        self.adaptive_feature.train(True)
        for i in range(100):
            opt.zero_grad()
            obs_feature = self.adaptive_feature(train_obs_data_t)
            loss = linear_reg_loss(train_latent_feature_t, obs_feature, self.reg2)
            loss.backward()
            opt.step()

        with torch.no_grad():
            self.adaptive_feature.train(False)
            train_obs_feature = self.adaptive_feature(train_obs_data_t).detach().numpy()
            test_obs_feature = self.adaptive_feature(test_obs_data_t).detach().numpy()

        obs_kernel_mat = train_obs_feature @ train_obs_feature.T
        obs_test_kernel_mat = train_obs_feature @ test_obs_feature.T
        return obs_kernel_mat, obs_test_kernel_mat

    def tune_stage1_reg(self):
        lam1_candi = [0.001, 0.005, 0.0001, 0.0005, 0.00005]
        scores = [cal_loocv_emb(self.latent_kernel_mat[:-1, :][:, :-1],
                                self.latent_kernel_mat[1:, :][:, 1:], lam) for lam in lam1_candi]
        self.reg1 = lam1_candi[np.argmin(scores)]

    def compute_filtering_original(self, gamma, test_kernel):
        n_data = self.train_length
        D = np.diag(gamma[:, 0]) * n_data
        A = D @ self.obs_kernel_mat
        weight = A @ np.linalg.solve(A @ A + self.reg2 * np.eye(n_data), D @ test_kernel)
        return weight

    def compute_filtering_proposed(self, gamma, test_kernel):
        n_data = self.train_length
        D = np.diag(np.maximum(gamma[:, 0], 0.0))
        D_sq = np.sqrt(D)
        weight = D_sq @ np.linalg.solve(D_sq @ self.obs_kernel_mat @ D_sq + self.reg2 * np.eye(n_data),
                                        D_sq @ test_kernel)
        return weight

    def fit_predict_matrix(self):
        n_data = self.train_length - 1
        K = (self.latent_kernel_mat[:-1, :][:, :-1]) + n_data * np.eye(n_data) * self.reg1
        self.predict_weight = np.linalg.solve(K, self.latent_kernel_mat[:-1, :])
        self.predict_weight = np.block([[np.zeros(n_data + 1)],
                                        [self.predict_weight]])  # shape: (T, T)

    def fit_kernel_bayes_stage1_matrix(self):
        n_data = self.train_length
        K = self.latent_kernel_mat + n_data * np.eye(n_data) * self.reg1
        self.kernel_bayes_stage1_matrix = np.linalg.solve(K, self.latent_kernel_mat)  # shape: (T, T)


def kernel_bayes_filter_experiment(env_param: Dict[str, Any],
                                   mdl_param: Dict[str, Any],
                                   dump_dir: Path,
                                   rand_seed: int,
                                   verbose: int):
    train_length = env_param["n_train"]
    test_length = env_param["n_test"]

    data = generate_sequence(length=train_length + test_length,
                             rand_seed=rand_seed,
                             options=env_param)
    adaptive_feature = None
    if mdl_param.get("use_adaptive_feature", False):
        adaptive_feature, _ = obtain_feature_model(env_param["name"])
    model = KBR(train_length=train_length, adaptive_feature=adaptive_feature, **mdl_param)
    try:
        return model.fit_and_test(data)
    except:
        return np.NaN
