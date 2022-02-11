from typing import Optional, Tuple
from torch import nn
from torch.nn.utils import spectral_norm


def obtain_feature_model(name: str) -> Tuple[nn.Module, int]:
    if name == "dsprite":
        feature_dim = 32
        feature = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(1024, 512)),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                spectral_norm(nn.Linear(512, 128)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(128, feature_dim)),
                                nn.ReLU())
    elif name == "maze":
        feature_dim = 32
        dropout = 0.5
        feature = nn.Sequential(spectral_norm(nn.Linear(36 * 48 * 3, 1024)),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(1024, 512)),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                spectral_norm(nn.Linear(512, 128)),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(128, feature_dim)),
                                nn.ReLU())

    elif name == "synthetic":
        feature_dim = 64
        feature = nn.Sequential(nn.Linear(2, 128),
                                nn.ReLU(),
                                nn.Linear(128, feature_dim),
                                nn.ReLU())

    else:
        raise ValueError(f"name {name} not known")

    return feature, feature_dim
