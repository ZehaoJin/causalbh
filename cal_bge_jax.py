# from Tristan

import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd

from jax.scipy.special import gammaln
from scipy.special import logsumexp
from tqdm.auto import trange


def _slogdet_jax(array, parents):
    """Log-determinant of a submatrix.

    This function is `jax.jit`-compatible and differentiable (`jax.grad`-compatible)
    by masking everything but the submatrix, and adding a diagonal of ones
    everywhere else to obtain the expected determinant. We assume that the input
    matrix is positive-definite (meaning that the result is guaranteed to be > 0).

    Inspired from: https://github.com/larslorch/dibs/blob/master/dibs/utils/func.py#L128

    Parameters
    ----------
    array : jnp.Array, shape `(num_variables, num_variables)`
        The matrix to compute the log-determinant of.

    parents : jnp.Array, shape `(num_variables,)`
        A binary vector containing the rows and columns of `array` to keep in
        the computation of the log-determinant of the submatrix.

    Returns
    -------
    logdet : jnp.Array, shape `()`
        The log-determinant of the submatrix, indexed by `parents` on both dimensions.
    """
    mask = jnp.outer(parents, parents)
    submat = mask * array + jnp.diag(1. - parents)
    return jnp.linalg.slogdet(submat)[1]


class BGe:
    def __init__(self, num_variables, mean_obs=None, alpha_mu=1., alpha_w=None):
        self.num_variables = num_variables
        self.mean_obs = mean_obs or jnp.zeros((num_variables,))
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w or (num_variables + 2.)

    def log_prob(self, observations, adjacency):
        """Compute the log-marginal probability log P(D | G).

        Parameters
        ----------
        observations : jnp.Array, shape `(num_observations, num_variables)`
            The dataset of observations D.
        
        adjacency : jnp.array, shape `(num_variables, num_variables)`
            The adjacency matrix G

        Returns
        -------
        log_prob : jnp.Array, shape `(num_variables,)`
            The log-marginal probability log P(X_i | Pa(X_i)) for each variable.
        """
        def _log_prob(target, observations, parents):
            num_parents = jnp.sum(parents)
            num_observations = observations.shape[0]

            t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)
            T = t * jnp.eye(self.num_variables)

            # covariance matrix of observational data entries
            data_mean = jnp.mean(observations, axis=0)
            data_centered = observations - data_mean

            R = (T + (data_centered.T @ data_centered)
                + ((num_observations * self.alpha_mu) / (num_observations + self.alpha_mu)) * \
                jnp.outer(data_mean - self.mean_obs, data_mean - self.mean_obs)
            )

            factor = self.alpha_w - self.num_variables + num_parents + 1
            log_gamma_term = (
                0.5 * (jnp.log(self.alpha_mu) - jnp.log(num_observations + self.alpha_mu))
                + gammaln(0.5 * (num_observations + factor)) - gammaln(0.5 * factor)
                - 0.5 * num_observations * jnp.log(jnp.pi)
                + 0.5 * (factor + num_parents) * jnp.log(t)
            )

            variables = parents.at[target].set(1)
            factor = num_observations + self.alpha_w - self.num_variables + num_parents
            log_term_r = 0.5 * (
                factor * _slogdet_jax(R, parents)
                - (factor + 1) * _slogdet_jax(R, variables)
            )

            return (log_gamma_term + log_term_r) * (num_observations > 0)

        targets = jnp.arange(self.num_variables)
        v_log_prob = jax.vmap(_log_prob, in_axes=(0, None, 1))
        return v_log_prob(targets, observations, adjacency)
    



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
    #from pathlib import Path

    folder = '/data/zj448/causal/exact_posteriors'
    n = 7


    galtypes = ['ell','len','spr']

    for galtype in galtypes:
        print(galtype)
        data_path = '/home/zj448/causalbh/R_e_data/causal_BH_'+galtype+'.csv'
        # load data
        data = pd.read_csv(data_path)
        data = (data - data.mean()) / data.std()  # Standardize data
        data = data.to_numpy()

        log_probs = compute_exact_posterior(
            os.path.join(folder, f'dags_{n}.npy'),
            data,
            batch_size=512,
            verbose=True
        )
        print(log_probs.shape)
        print(logsumexp(log_probs))

        # Save the exact posteriors
        np.save(folder+'/bge_'+galtype+'.npy', log_probs)