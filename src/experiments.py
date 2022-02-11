from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed
import logging

from src.utils import grid_search_dict
from src.model.extended_kalman_filter import extended_kalman_filter_experiment
from src.model.kernel_bayes_filter import kernel_bayes_filter_experiment
from src.model.rnn import rnn_experiment
from src.model.nn import nn_experiment

logger = logging.getLogger()


def get_run_func(mdl_name: str):
    if mdl_name == "EKF":
        return extended_kalman_filter_experiment
    elif mdl_name == "KBF":
        return kernel_bayes_filter_experiment
    elif mdl_name == "RNN":
        return rnn_experiment
    elif mdl_name == "NN":
        return nn_experiment
    else:
        raise ValueError(f"name {mdl_name} is not known")


def experiments(configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int):

    data_config = configs["data"]
    model_config = configs["model"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    run_func = get_run_func(model_config["name"])
    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir
            tasks = [delayed(run_func)(env_param, mdl_param, one_mdl_dump_dir, idx, verbose) for idx in range(n_repeat)]
            res = Parallel(n_jobs=num_cpus)(tasks)
            np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(res))
        logger.critical(f"{dump_name} ended")
