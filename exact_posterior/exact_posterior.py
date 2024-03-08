import numpy as np
import jax.numpy as jnp
import jax

from scipy.special import logsumexp
from tqdm.auto import trange

from bge_score_jax import BGe


def compute_exact_posterior(filename, observations, batch_size=1, verbose=True):
    num_variables = observations.shape[1]
    model = BGe(num_variables=num_variables)

    @jax.jit
    def log_prob(observations, adjacencies):
        v_log_prob = jax.vmap(model.log_prob, in_axes=(None, 0))
        log_probs = v_log_prob(observations, adjacencies)
        return jnp.sum(log_probs, axis=1)

    # Load all the DAGs
    with open(filename, 'rb') as f:
        dags_compressed = np.load(f)
    num_dags = dags_compressed.shape[0]

    log_probs = np.zeros((num_dags,), dtype=np.float32)
    for i in trange(0, len(dags_compressed), batch_size, disable=(not verbose)):
        # Get a batch of DAGs & uncompress
        batch_compressed = dags_compressed[i:i + batch_size]
        batch = np.unpackbits(batch_compressed, axis=1, count=num_variables ** 2)
        batch = batch.reshape(-1, num_variables, num_variables)

        # Compute the BGe scores
        log_probs[i:i + batch_size] = log_prob(observations, batch)

    # Normalize the log-marginal probabilities
    log_probs = log_probs - logsumexp(log_probs)
    return log_probs


if __name__ == '__main__':
    import os
    from pathlib import Path

    # Random data
    observations = np.random.normal(size=(100, 6))

    root = Path(os.getenv('SLURM_TMPDIR'))
    log_probs = compute_exact_posterior(
        root / 'dags.npy',
        observations,
        batch_size=512,
        verbose=True
    )
    print(log_probs.shape)
    print(logsumexp(log_probs))
