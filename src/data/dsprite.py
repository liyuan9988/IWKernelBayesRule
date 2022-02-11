import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from itertools import product
import pathlib

from src.data.data_class import TimeSequence
from src.data.synthetic import compile_dynamics

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")


def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    orientation_id_arr = np.array([0] * data_size, dtype=int)
    scale_id_arr = np.array([0] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)


def generate_dsprite_sequence(length: int, rand_seed: int, eta: float, beta: float,
                              freq: int, latent_noise: float, obs_noise: float, image_noise: float,
                              **kwargs):

    rng = default_rng(seed=rand_seed)
    state = np.array([1.0, 0.0])
    dynamics, _ = compile_dynamics(eta, beta, freq)
    latent = []
    noise_mat = []
    obs_latent = []
    for i in range(length):
        state = np.array(dynamics(state))
        state = state + rng.normal(scale=latent_noise, size=state.shape)
        latent.append(state)
        obs_latent.append(state + rng.normal(scale=obs_noise, size=state.shape))
        noise_mat.append(rng.normal(scale=image_noise, size=64*64))

    latent = np.array(latent)
    noise_mat = np.array(noise_mat)
    obs_latent = np.array(obs_latent)

    # compile dSprite
    dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                          allow_pickle=True, encoding="bytes")
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    points = np.linspace(0.0, 1.0, 32)[:, np.newaxis] * 3 - 1.5
    X_arr = np.argmin(cdist(points, obs_latent[:, [0]]), axis=0)
    Y_arr = np.argmin(cdist(points, obs_latent[:, [1]]), axis=0)
    image_idx_arr = image_id(latents_bases, X_arr, Y_arr)
    obs = imgs[image_idx_arr].reshape((length, 64 * 64)).astype(np.float32)
    obs += noise_mat

    return TimeSequence(latent=latent,
                        obs=obs)
