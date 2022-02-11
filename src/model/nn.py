import numpy as np
from typing import Dict, Any
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam

from src.data import generate_sequence
from src.data.data_class import TimeSequence
from src.feature_map import obtain_feature_model
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred

USE_LR = False


def nn_experiment(env_param: Dict[str, Any],
                   mdl_param: Dict[str, Any],
                   dump_dir: Path,
                   rand_seed: int,
                   verbose: int):
    # options
    train_length = env_param["n_train"]
    test_length = env_param["n_test"]
    data_name = env_param["name"]

    reg = mdl_param["reg"]
    n_iter = mdl_param["n_iter"]


    whole_seq = generate_sequence(length=train_length + test_length,
                                  rand_seed=rand_seed,
                                  options=env_param)

    train_seq = TimeSequence(latent=whole_seq.latent[:train_length],
                             obs=whole_seq.obs[:train_length])

    test_seq = TimeSequence(latent=whole_seq.latent[train_length:],
                            obs=whole_seq.obs[train_length:])

    train_data, train_target = train_seq.obs, train_seq.latent
    test_data, test_target = test_seq.obs, test_seq.latent

    train_data_t = torch.tensor(train_data, dtype=torch.float32)
    train_target_t = torch.tensor(train_target, dtype=torch.float32)
    test_data_t = torch.tensor(test_data, dtype=torch.float32)

    if USE_LR:
      mdl, feature_dim = obtain_feature_model(data_name)
      opt = Adam(mdl.parameters())
    else:
      mdl, feature_dim = obtain_feature_model(data_name)
      target_dim = train_target.shape[1]
      output_layer = nn.Linear(feature_dim, target_dim)
      opt = Adam(list(mdl.parameters()) + list(output_layer.parameters()))
      criterion = nn.MSELoss()

    loss_list = []
    test_loss_list = []

    mdl.train(True)
    for i in range(n_iter):
        opt.zero_grad()
        feature = mdl(train_data_t)

        if USE_LR:
          loss = linear_reg_loss(train_target_t, feature, reg)
        else:
          pred = output_layer(feature)
          loss = criterion(pred, train_target_t)

        loss_list.append(loss.detach().numpy())

        loss.backward()
        opt.step()

        mdl.train(False)
        if USE_LR:
          feature = mdl(train_data_t)
          weight = fit_linear(train_target_t, feature, reg)
          pred = linear_reg_pred(mdl(test_data_t), weight).detach().numpy()
        else:
          feature = mdl(test_data_t)
          pred = output_layer(feature).detach().numpy()
        test_loss_list.append(np.sum((pred - test_target) ** 2) / test_target.shape[0])
        mdl.train(True)


    loss_list = np.array(loss_list)
    test_loss_list = np.array(test_loss_list)

    mdl.train(False)
    if USE_LR:
      feature = mdl(train_data_t)
      weight = fit_linear(train_target_t, feature, reg)
      pred = linear_reg_pred(mdl(test_data_t), weight).detach().numpy()
    else:
      feature = mdl(test_data_t)
      pred = output_layer(feature).detach().numpy()

    final_err = np.sum((pred - test_target) ** 2) / test_target.shape[0]
    # return np.concatenate([[final_err], loss_list, test_loss_list])

    final_err_std = np.sum((pred - test_target) ** 2, 1).std()
    return np.concatenate([[final_err, final_err_std], loss_list, test_loss_list])
