import numpy as np
from numpy.random import default_rng
import pickle
from skimage import io

from skimage.transform import resize
from scipy.spatial.distance import cdist

import pathlib

from src.data.data_class import TimeSequence
from src.data.synthetic import compile_dynamics

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/rotate_left_to_right/")


def obtain_img(latent, img_list_arr, img_info_arr):
    latent_degree = latent / np.pi * 180
    latent = latent_degree - (latent_degree // 360) * 360 - 180
    dist = np.abs(img_info_arr - latent)
    img = img_list_arr[np.argmin(dist)]
    image_resized = resize(img, (36, 48), anti_aliasing=True).ravel()
    return image_resized


def generate_maze_sequence(length: int, rand_seed: int, rotation: float,
                           latent_noise: float, obs_noise: float, image_noise: float, **kwargs):
    rng = default_rng(seed=rand_seed)
    state = 0.0
    latent = []
    obs = []
    data = pickle.load(open(DATA_PATH.joinpath("nav_maze_static_01/seed_123.pkl"), "rb"))
    img_list = data["image"]
    img_info = data["position"][:, 1]

    for i in range(length):
        state = state + rng.normal(loc=rotation, scale=latent_noise)
        latent.append([np.cos(state), np.sin(state)])
        obs_latent = state + rng.normal(loc=0.0, scale=obs_noise)
        img = obtain_img(obs_latent, img_list, img_info)
        obs.append(img + rng.normal(scale=image_noise, size=img.shape))

    latent = np.array(latent)
    obs = np.array(obs)
    return TimeSequence(latent=latent, obs=obs)
