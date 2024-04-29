import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import os
import pandas as pd
import math
import seaborn as sns

from scipy.special import logsumexp
from pathlib import Path
from tqdm.auto import tqdm, trange

from bge_score_jax import BGe


def compute_exact_posterior(dags_compressed, observations, batch_size=1, verbose=True):
    num_variables = observations.shape[1]
    model = BGe(num_variables=num_variables)

    @jax.jit
    def log_prob(observations, adjacencies_compressed):
        adjacencies = jnp.unpackbits(adjacencies_compressed, axis=1, count=num_variables ** 2)
        adjacencies = adjacencies.reshape(-1, num_variables, num_variables)

        v_log_prob = jax.vmap(model.log_prob, in_axes=(None, 0))
        log_probs = v_log_prob(observations, adjacencies)
        return jnp.sum(log_probs, axis=1)

    num_dags = dags_compressed.shape[0]
    log_probs = np.zeros((num_dags,), dtype=np.float32)
    for i in trange(0, num_dags, batch_size, disable=(not verbose)):
        # Get a batch of (compressed) DAGs
        batch_compressed = dags_compressed[i:i + batch_size]

        # Compute the BGe scores
        log_probs[i:i + batch_size] = log_prob(observations, batch_compressed)

    # Normalize the log-marginal probabilities
    log_probs = log_probs - logsumexp(log_probs)
    return log_probs


def edge_log_marginal(dags_compressed, log_joint, num_variables, batch_size=1, verbose=True):
    @jax.jit
    def marginalize(log_probs, adjacencies_compressed):
        adjacencies = jnp.unpackbits(adjacencies_compressed, axis=1, count=num_variables ** 2)
        log_probs = jnp.where(adjacencies == 1, log_probs[:, None], -jnp.inf)
        return jax.nn.logsumexp(log_probs, axis=0)

    num_dags = dags_compressed.shape[0]
    log_marginal = []
    for i in trange(0, num_dags, batch_size, disable=(not verbose)):
        # Get a batch of data
        batch_compressed = dags_compressed[i:i + batch_size]
        log_probs = log_joint[i:i + batch_size]

        log_marginal.append(marginalize(log_probs, batch_compressed))

    log_marginal = np.stack(log_marginal, axis=0)
    log_marginal = logsumexp(log_marginal, axis=0)
    return log_marginal.reshape(num_variables, num_variables)


def get_transitive_closure(adjacency):
    # Warshall's algorithm
    def scan_fun(closure, i):
        outer_product = jnp.outer(closure[:, i], closure[i])
        return (jnp.logical_or(closure, outer_product), None)
    
    adjacency = adjacency.astype(jnp.bool_)
    arange = jnp.arange(adjacency.shape[0])
    closure, _ = jax.lax.scan(scan_fun, adjacency, arange)

    return closure

def path_log_marginal(dags_compressed, log_joint, num_variables, batch_size=1, verbose=True):
    @jax.jit
    def marginalize(log_probs, adjacencies_compressed):
        adjacencies = jnp.unpackbits(adjacencies_compressed, axis=1, count=num_variables ** 2)
        adjacencies = adjacencies.reshape(-1, num_variables, num_variables)
        closures = jax.vmap(get_transitive_closure)(adjacencies)
        log_probs = jnp.where(closures, log_probs[:, None, None], -jnp.inf)
        return jax.nn.logsumexp(log_probs, axis=0)

    num_dags = dags_compressed.shape[0]
    log_marginal = []
    for i in trange(0, num_dags, batch_size, disable=(not verbose)):
        # Get a batch of data
        batch_compressed = dags_compressed[i:i + batch_size]
        log_probs = log_joint[i:i + batch_size]

        log_marginal.append(marginalize(log_probs, batch_compressed))

    log_marginal = np.stack(log_marginal, axis=0)
    log_marginal = logsumexp(log_marginal, axis=0)
    return log_marginal.reshape(num_variables, num_variables)


######
dags_compressed = np.load('exact_posteriors/dags_7.npy')
dags_compressed.shape
max_iter = 100
root = Path('/home/zj448/causal/jax-dag-gflownet/exact_posteriors')


datasets = ['ell', 'len', 'spr']
dfs, observations = {}, {}

keys=['M_BH','log_sigma0','log_R_e_sph_eq_kpc','log<Sigma>_e','GJC23log(M*,gal/M_sun)','GJC23log(sSFR)','GJC23W2-W3']
std_keys=['M_BH_std_sym','log_sigma0_std','log_R_e_sph_eq_kpc_std','log<Sigma>_e_std','GJC23log(M*,gal/M_sun)_std','GJC23log(sSFR)_std','GJC23W2-W3_std']


# [iter,ell/len/spr,keys,keys]
edge_marginal_array=np.zeros((max_iter,len(datasets),len(keys),len(keys)))
path_marginal_array=np.zeros((max_iter,len(datasets),len(keys),len(keys)))

for seed in tqdm(range(max_iter),total=max_iter):
    np.random.seed(seed)
    print('running',seed,'/',max_iter)
    for dataset in datasets:
        dfs[dataset] = pd.read_csv(f'R_e_data/causal_BH_{dataset}_std.csv')
        # sampling from 1 sigma observation error
        df_sampled=pd.DataFrame()
        for key,std_key in zip(keys,std_keys):
            df_sampled[key]=dfs[dataset][key]+np.random.randn(len(dfs[dataset]))*dfs[dataset][std_key]
        dfs[dataset] = df_sampled.copy()
        # standardize data
        dfs[dataset] = dfs[dataset].apply(lambda col: ( col - col.mean() ) / col.std(), axis=0)  # Standardize data
        observations[dataset] = np.asarray(dfs[dataset])

    # calculate log joint probability, and calculate edge marginals
    log_edge_marginals = {}
    log_joint = {}
    for dataset in datasets:
        log_joint[dataset] = compute_exact_posterior(dags_compressed, observations[dataset], batch_size=2048)
        log_edge_marginals[dataset] = edge_log_marginal(dags_compressed,
            log_joint[dataset], observations[dataset].shape[1], batch_size=4096)
        

    # store edge marginals
    for i, dataset in enumerate(datasets):
        edge_marginal_array[seed,i,:,:]=np.exp(log_edge_marginals[dataset])

    # calculate path marginals
    log_path_marginals = {}
    for dataset in datasets:
        log_path_marginals[dataset] = path_log_marginal(dags_compressed,
            log_joint[dataset], observations[dataset].shape[1], batch_size=1024)
        
    for i, dataset in enumerate(datasets):
        path_marginal_array[seed,i,:,:]=np.exp(log_path_marginals[dataset])

    
    np.save('edge_marginal_std.npy',edge_marginal_array)    
    np.save('path_marginal_std.npy',path_marginal_array)    
