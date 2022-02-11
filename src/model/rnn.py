import numpy as np
from typing import Dict, Any
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam

from src.data import generate_sequence
from src.data.data_class import TimeSequence
from src.feature_map import obtain_feature_model


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim: int, feature_extractor: nn.Module, feature_dim: int, target_dim: int, **kwargs):
        super(LSTMModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.rnn = nn.LSTM(input_size=feature_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, target_dim)

    def forward(self, inputs: nn.Module, hidden0=None):
        nbatch, nfilter, ndim = inputs.shape
        feature_table = self.feature_extractor(inputs.reshape(nbatch * nfilter, ndim))
        feature_table = feature_table.reshape(nbatch, nfilter, self.feature_dim)
        output, (hidden, cell) = self.rnn(feature_table, hidden0)
        return self.output_layer(output[:, -1, :])


def compile_data_for_rnn(whole_seq: TimeSequence, n_filter: int):
    target = whole_seq.latent[n_filter - 1:, :]
    n_data = whole_seq.obs.shape[0] - (n_filter - 1)
    data = [(whole_seq.obs[i:i + n_filter])[np.newaxis, :, :] for i in range(n_data)]
    return np.concatenate(data, axis=0), target


def rnn_experiment(env_param: Dict[str, Any],
                   mdl_param: Dict[str, Any],
                   dump_dir: Path,
                   rand_seed: int,
                   verbose: int):
    if mdl_param.get("earlystop", False):
        return rnn_experiment_with_earlystop(env_param,
                                             mdl_param,
                                             dump_dir,
                                             rand_seed,
                                             verbose)
    else:
        return rnn_experiment_without_earlystop(env_param,
                                                mdl_param,
                                                dump_dir,
                                                rand_seed,
                                                verbose)


def rnn_experiment_without_earlystop(env_param: Dict[str, Any],
                                     mdl_param: Dict[str, Any],
                                     dump_dir: Path,
                                     rand_seed: int,
                                     verbose: int):
    # options
    train_length = env_param["n_train"]
    test_length = env_param["n_test"]
    data_name = env_param["name"]

    n_filter = mdl_param["n_filter"]
    n_iter = mdl_param["n_iter"]

    whole_seq = generate_sequence(length=train_length + test_length,
                                  rand_seed=rand_seed,
                                  options=env_param)

    train_seq = TimeSequence(latent=whole_seq.latent[:train_length],
                             obs=whole_seq.obs[:train_length])

    test_seq = TimeSequence(latent=whole_seq.latent[train_length - (n_filter - 1):],
                            obs=whole_seq.obs[train_length - (n_filter - 1):])

    train_data, train_target = compile_data_for_rnn(train_seq, n_filter)
    test_data, test_target = compile_data_for_rnn(test_seq, n_filter)

    train_data_t = torch.tensor(train_data, dtype=torch.float32)
    train_target_t = torch.tensor(train_target, dtype=torch.float32)
    test_data_t = torch.tensor(test_data, dtype=torch.float32)

    adaptive_feature, feature_dim = obtain_feature_model(data_name)
    target_dim = train_target.shape[1]
    mdl = LSTMModel(feature_extractor=adaptive_feature,
                    feature_dim=feature_dim,
                    target_dim=target_dim,
                    **mdl_param)
    opt = Adam(mdl.parameters())
    criterion = nn.MSELoss()
    mdl.train(True)
    for i in range(n_iter):
        opt.zero_grad()
        pred = mdl(train_data_t)
        loss = criterion(pred, train_target_t)
        loss.backward()
        opt.step()

    mdl.train(False)
    pred = mdl(test_data_t).detach().numpy()
    return np.sum((pred - test_target) ** 2) / test_target.shape[0]


def rnn_experiment_with_earlystop(env_param: Dict[str, Any],
                                  mdl_param: Dict[str, Any],
                                  dump_dir: Path,
                                  rand_seed: int,
                                  verbose: int):
    # options
    train_length = env_param["n_train"]
    test_length = env_param["n_test"]
    data_name = env_param["name"]

    n_filter = mdl_param["n_filter"]
    n_iter = mdl_param["n_iter"]

    whole_seq = generate_sequence(length=train_length + test_length,
                                  rand_seed=rand_seed,
                                  options=env_param)

    # Sub-sequence for training. The remaining is for validation.
    train_train_length = int(train_length * 0.9)
    train_seq = TimeSequence(latent=whole_seq.latent[:train_train_length],
                             obs=whole_seq.obs[:train_train_length])
    eval_seq = TimeSequence(
        latent=whole_seq.latent[train_train_length - (n_filter - 1):train_length],
        obs=whole_seq.obs[train_train_length - (n_filter - 1):train_length])

    test_seq = TimeSequence(latent=whole_seq.latent[train_length - (n_filter - 1):],
                            obs=whole_seq.obs[train_length - (n_filter - 1):])

    train_data, train_target = compile_data_for_rnn(train_seq, n_filter)
    eval_data, eval_target = compile_data_for_rnn(eval_seq, n_filter)
    test_data, test_target = compile_data_for_rnn(test_seq, n_filter)

    train_data_t = torch.tensor(train_data, dtype=torch.float32)
    train_target_t = torch.tensor(train_target, dtype=torch.float32)
    eval_data_t = torch.tensor(eval_data, dtype=torch.float32)
    test_data_t = torch.tensor(test_data, dtype=torch.float32)

    adaptive_feature, feature_dim = obtain_feature_model(data_name)
    target_dim = train_target.shape[1]
    mdl = LSTMModel(feature_extractor=adaptive_feature,
                    feature_dim=feature_dim,
                    target_dim=target_dim,
                    **mdl_param)
    opt = Adam(mdl.parameters())
    criterion = nn.MSELoss()

    loss_list = []
    eval_loss_list = []
    test_loss_list = []

    mdl.train(True)
    for i in range(n_iter):
        opt.zero_grad()
        pred = mdl(train_data_t)
        loss = criterion(pred, train_target_t)
        loss_list.append(loss.detach().numpy())
        loss.backward()
        opt.step()

        mdl.train(False)
        pred = mdl(eval_data_t).detach().numpy()
        eval_loss_list.append(np.sum((pred - eval_target) ** 2) / eval_target.shape[0])
        pred = mdl(test_data_t).detach().numpy()
        test_loss_list.append(np.sum((pred - test_target) ** 2) / test_target.shape[0])
        mdl.train(True)

    loss_list = np.array(loss_list)
    eval_loss_list = np.array(eval_loss_list)
    test_loss_list = np.array(test_loss_list)

    return test_loss_list[np.argmin(eval_loss_list)]
