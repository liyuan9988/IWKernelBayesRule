import numpy as np
from typing import Dict, Any
from pathlib import Path
from filterpy.kalman import ExtendedKalmanFilter

from src.data.synthetic import compile_dynamics, generate_synthetic_sequence


def extended_kalman_filter_experiment(env_param: Dict[str, Any],
                                      mdl_param: Dict[str, Any],
                                      dump_dir: Path,
                                      rand_seed: int,
                                      verbose: int):
    # options
    assert env_param["name"] == "synthetic"
    eta = env_param["eta"]
    beta = env_param["beta"]
    freq = env_param["freq"]
    latent_noise = env_param["latent_noise"]
    obs_noise = env_param["obs_noise"]
    seq_length = env_param["n_test"]

    rk = ExtendedKalmanFilter(dim_x=2, dim_z=2)
    rk.x = np.array([1.0, 0.0])
    rk.Q = np.eye(2) * (latent_noise ** 2)
    rk.R = np.eye(2) * (obs_noise ** 2)
    rk.P *= 50

    data = generate_synthetic_sequence(length=seq_length, rand_seed=rand_seed, eta=eta, beta=beta, freq=freq,
                                       latent_noise=latent_noise, obs_noise=obs_noise)
    dynamics, jacobian_func = compile_dynamics(eta=eta, beta=beta, freq=freq)

    pred = []
    for i in range(seq_length):
        new_obs = data.obs[i, :]
        rk.update(new_obs, lambda x: np.eye(2), lambda x: x)
        pred.append(rk.x)

        # implement predict step
        next_latent = dynamics(rk.x)
        jacobian = jacobian_func(rk.x)
        rk.x = next_latent
        rk.P = jacobian @ rk.P @ jacobian.T + rk.Q

    return np.sum((np.array(pred) - data.latent) ** 2) / data.latent.shape[0]
