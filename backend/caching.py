import time
from itertools import product
from pathlib import Path
from joblib import Memory
from math import prod
import numpy as np
from dp_fl_simulation import run_dp_federated_learning   # noqa: E402
from tqdm import tqdm
CACHE_DIR = Path("cache_results")
memory = Memory(CACHE_DIR, verbose=0)
run_dp_federated_learning = memory.cache(run_dp_federated_learning)

PARAM_GRID = {
    "epsilon":      np.round(np.arange(1, 10, 0.5), 2).tolist(),
    "num_clients":  [i for i in range(5, 50, 5)],
    "rounds":       [i for i in range(5, 100, 10)],
    "dp_noise":     [True, False],
}

MAX_MINUTES = 30                          # stop after this wall‑clock time


def main():
    combos = product(
        PARAM_GRID["epsilon"],
        PARAM_GRID["num_clients"],
        PARAM_GRID["rounds"],
        PARAM_GRID["dp_noise"],
    )

    for eps, n_cli, n_rounds, dp in tqdm(combos, total=prod(len(v) for v in PARAM_GRID.values())):
        run_dp_federated_learning(
            epsilon=eps,
            clip=1,
            num_clients=n_cli,
            mechanism="Gaussian",
            rounds=n_rounds,
            dp_noise=dp,
        )

if __name__ == "__main__":
    main()
