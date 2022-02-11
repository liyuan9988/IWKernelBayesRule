import numpy as np
import sympy
from numpy.random import default_rng

from src.data.data_class import TimeSequence


def compile_dynamics(eta: float, beta: float, freq: int):
    ut = sympy.Symbol("ut")
    vt = sympy.Symbol("vt")

    ct = ut / sympy.sqrt(ut ** 2 + vt ** 2)
    st = vt / sympy.sqrt(ut ** 2 + vt ** 2)

    next_ct = np.cos(eta) * ct - np.sin(eta) * st
    next_st = np.sin(eta) * ct + np.cos(eta) * st

    rotate_map = sympy.Matrix([[next_ct, -next_st],
                               [next_st, next_ct]])
    coef = beta * (rotate_map ** freq)[1, 0]

    dynamics = sympy.lambdify([[ut, vt]], [next_ct * (1 + coef), next_st * (1 + coef)], "numpy")

    jacob_func = sympy.Matrix([next_ct * (1 + coef), next_st * (1 + coef)]).jacobian(sympy.Matrix([ut, vt]))
    dynamics_jacob = sympy.lambdify([[ut, vt]], jacob_func, "numpy")
    return dynamics, dynamics_jacob


def generate_synthetic_sequence(length: int, rand_seed: int, eta: float, beta: float,
                                freq: int, latent_noise: float, obs_noise: float, **kwargs) -> TimeSequence:
    rng = default_rng(rand_seed)
    latent = []
    obs = []
    state = np.array([1.0, 0.0])
    dynamics, _ = compile_dynamics(eta, beta, freq)
    for i in range(length):
        state = np.array(dynamics(state))
        state = state + rng.normal(scale=latent_noise, size=state.shape)
        latent.append(state)
        obs_one = state + rng.normal(scale=obs_noise, size=state.shape)
        obs.append(obs_one)

    return TimeSequence(latent=np.array(latent),
                        obs=np.array(obs))

