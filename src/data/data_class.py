import numpy as np
from typing import Optional, NamedTuple


class TimeSequence(NamedTuple):
    latent: np.ndarray  # [length, dimension]
    obs: np.ndarray  # [length, dimension]

