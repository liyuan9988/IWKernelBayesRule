from typing import Dict, Any

from .data_class import TimeSequence
from .synthetic import generate_synthetic_sequence
from .dsprite import generate_dsprite_sequence
from .maze import generate_maze_sequence


def generate_sequence(length: int, rand_seed: int, options: Dict[str, Any]):
    name = options["name"]
    if name == "synthetic":
        return generate_synthetic_sequence(length=length, rand_seed=rand_seed, **options)
    elif name == "dsprite":
        return generate_dsprite_sequence(length=length, rand_seed=rand_seed, **options)
    elif name == "maze":
        return generate_maze_sequence(length=length, rand_seed=rand_seed, **options)
    else:
        raise ValueError(f"Dataset {name} not known")