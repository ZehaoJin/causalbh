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
    def log_prob(observations, adjacencies_compressed):
        adjacencies = jnp.unpackbits(adjacencies_compressed, axis=1, count=num_variables ** 2)
        adjacencies = adjacencies.reshape(-1, num_variables, num_variables)

        v_log_prob = jax.vmap(model.log_prob, in_axes=(None, 0))
        log_probs = v_log_prob(observations, adjacencies)
        return jnp.sum(log_probs, axis=1)

    # Load all the DAGs
    with open(filename, 'rb') as f:
        dags_compressed = np.load(f)
    num_dags = dags_compressed.shape[0]

    log_probs = np.zeros((num_dags,), dtype=np.float32)
    for i in trange(0, len(dags_compressed), batch_size, disable=(not verbose)):
        # Get a batch of (compressed) DAGs
        batch_compressed = dags_compressed[i:i + batch_size]

        # Compute the BGe scores
        log_probs[i:i + batch_size] = log_prob(observations, batch_compressed)

    # Normalize the log-marginal probabilities
    log_probs = log_probs - logsumexp(log_probs)
    return log_probs


def compute_log_edge_marginal(filename, observations, batch_size=1, verbose=True):
    num_variables = observations.shape[1]
    model = BGe(num_variables=num_variables)

    @jax.jit
    def log_prob(observations, adjacencies_compressed):
        adjacencies_ = jnp.unpackbits(adjacencies_compressed, axis=1, count=num_variables ** 2)
        adjacencies = adjacencies_.reshape(-1, num_variables, num_variables)

        v_log_prob = jax.vmap(model.log_prob, in_axes=(None, 0))
        log_probs = v_log_prob(observations, adjacencies)
        log_probs = jnp.sum(log_probs, axis=1)

        log_probs_edges = jnp.where(adjacencies_, log_probs[:, None], -jnp.inf)
        return jax.nn.logsumexp(log_probs_edges, axis=0)

    # Load all the DAGs
    with open(filename, 'rb') as f:
        dags_compressed = np.load(f)
    num_dags = dags_compressed.shape[0]

    log_probs_marginals = []
    for i in trange(0, len(dags_compressed), batch_size, disable=(not verbose)):
        # Get a batch of (compressed) DAGs
        batch_compressed = dags_compressed[i:i + batch_size]

        # Compute the BGe scores
        log_probs_marginals.append(log_prob(observations, batch_compressed))

    # Normalize the log-marginal probabilities
    log_probs_marginals = np.stack(log_probs_marginals, axis=0)
    log_probs_marginals = logsumexp(log_probs_marginals, axis=0) - np.log(num_dags)
    return log_probs_marginals.reshape(num_variables, num_variables)


if __name__ == '__main__':
    import os
    from pathlib import Path

    # Random data
    observations = np.random.normal(size=(35, 7))

    root = Path(os.getenv('SLURM_TMPDIR'))
    log_probs = compute_log_edge_marginal(
        root / 'dags_7_final.npy',
        observations,
        batch_size=2048,
        verbose=True
    )

    with open(root / 'log_probs_edges.npy', 'wb') as f:
        np.save(f, log_probs)
