# From @Tristan

import numpy as np
from tqdm import tqdm
import math

from itertools import product, permutations


def all_dags_compressed(num_variables):
    """Generate all the DAGs over d variables.

    Parameters
    ----------
    num_variables : int
        The number of variables d in each DAG.

    Returns
    -------
    dags_compressed : np.ndarray, shape `(num_dags, ceil(num_variables ** 2 / 8))`
        The compressed representation of each DAG. Each row of this matrix
        represents a DAG, that has been flattened and compressed using
        `np.packbits` (i.e., treating a binary vector as a vector of bytes).
    """
    # Generate all the DAGs over num_variables nodes
    shape = (num_variables, num_variables)
    repeat = num_variables * (num_variables - 1) // 2

    # Generate all the possible binary codes
    codes = list(product([0, 1], repeat=repeat))
    codes = np.asarray(codes)

    # Get upper-triangular indices
    x, y = np.triu_indices(num_variables, k=1)

    # Fill the upper-triangular matrices
    trius = np.zeros((len(codes),) + shape, dtype=np.int_)
    trius[:, x, y] = codes

    # Apply permutation, and remove duplicates
    compressed_dags = set()
    for perm in tqdm(permutations(range(num_variables)),total=math.factorial(num_variables)):
        permuted = trius[:, :, perm][:, perm, :]
        permuted = permuted.reshape(-1, num_variables ** 2)
        permuted = np.packbits(permuted, axis=1)
        compressed_dags.update(map(tuple, permuted))
    compressed_dags = sorted(list(compressed_dags))

    return np.asarray(compressed_dags)


if __name__ == '__main__':
    import os
    #from pathlib import Path

    #root = Path(os.getenv('SLURM_TMPDIR'))
    folder = '/data/zj448/causal/exact_posteriors'
    n = 4


    dags_compressed = all_dags_compressed(num_variables=n)
    # with open(root / 'dags.npy', 'wb') as f:
    #     np.save(f, dags_compressed)
    
    # save as dags_n.npy
    np.save(os.path.join(folder, f'dags_{n}.npy'), dags_compressed)