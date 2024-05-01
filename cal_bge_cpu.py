import sys

from collections import namedtuple
from abc import ABC, abstractmethod
import math
import numpy as np
import networkx as nx
from scipy.special import gammaln
from functools import lru_cache
from tqdm import tqdm
from itertools import combinations
import pandas as pd

# cache to avoid recomputing the same local scores, making using of the modularity of BGe score
cache_max_size=10_000

# path to your data
data = pd.read_csv('your_folder/your_observational_data.csv')
dag_path = 'your_folder/your_saved_dags.npy'
score_path = 'your_folder/bge_score_output.npy'





LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])
data = (data - data.mean()) / data.std()  # Standardize data


class BaseScore(ABC):
    """Base class for the scorer.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_variables = len(self.column_names)
        self.prior.num_variables = self.num_variables
        self.column_names_to_idx = {name: idx for idx, name in enumerate(self.column_names)}
        self._cache_local_scores = None

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @abstractmethod
    def get_local_scores(self, target, indices, indices_after=None):
        pass

    @property
    def cache_local_scores(self):
        if self._cache_local_scores is None:
            self._cache_local_scores = lru_cache(cache_max_size)(self.get_local_scores)
        return self._cache_local_scores

    def score(self, graph):
        graph = nx.relabel_nodes(graph, self.column_names_to_idx)
        score = 0
        for node in graph.nodes():
            _, local_score = self.cache_local_scores(
                node, tuple(graph.predecessors(node)))
            score += local_score.score + local_score.prior
        return score
    

class BasePrior(ABC):
    """Base class for the prior over graphs p(G).
    
    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.
    
    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value





def logdet(array):
    _, logdet = np.linalg.slogdet(array)
    return logdet


class BGeScore(BaseScore):
    r"""BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one variable. The dataset D is assumed to only
        contain observational data (a `INT` column will be treated as
        a continuous variable like any other).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    mean_obs : np.ndarray (optional)
        Mean parameter of the Normal prior over the mean $\mu$. This array must
        have size `(N,)`, where `N` is the number of variables. By default,
        the mean parameter is 0.

    alpha_mu : float (default: 1.)
        Parameter $\alpha_{\mu}$ corresponding to the precision parameter
        of the Normal prior over the mean $\mu$.

    alpha_w : float (optional)
        Parameter $\alpha_{w}$ corresponding to the number of degrees of
        freedom of the Wishart prior of the precision matrix $W$. This
        parameter must satisfy `alpha_w > N - 1`, where `N` is the number
        of varaibles. By default, `alpha_w = N + 2`.
    """
    def __init__(
            self,
            data,
            prior,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None
        ):
        num_variables = len(data.columns)
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        super().__init__(data, prior)
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data = np.asarray(self.data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, target, indices):
        num_parents = len(indices)

        if indices:
            variables = [target] + list(indices)

            log_term_r = (
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                * logdet(self.R[np.ix_(indices, indices)])
                - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                * logdet(self.R[np.ix_(variables, variables)])
            )
        else:
            log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                * np.log(np.abs(self.R[target, target])))

        return LocalScore(
            key=(target, tuple(indices)),
            score=self.log_gamma_term[num_parents] + log_term_r,
            prior=self.prior(num_parents)
        )

    def get_local_scores(self, target, indices, indices_after=None):
        all_indices = indices if (indices_after is None) else indices_after
        local_score_after = self.local_score(target, all_indices)
        if indices_after is not None:
            local_score_before = self.local_score(target, indices)
        else:
            local_score_before = None
        return (local_score_before, local_score_after)


class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_variables,))
        return self._log_prior





bge = BGeScore(data, UniformPrior())
correct_num_dags=[1, 1, 3, 25, 543, 29281, 3781503, 1138779265]
n = len(data.columns)
all_dags=np.load(dag_path)

# check if the number of dags is correct
if len(all_dags) != correct_num_dags[n]:
    raise ValueError("The number of dags is incorrect")

# place holder for BGe scores
bge_scores = np.zeros(correct_num_dags[n])


# make a networkx graph
G = nx.DiGraph()
G.add_nodes_from(range(n))  # Add n nodes to the graph

# calculate BGe scores
for ind, adj_mat in enumerate(all_dags):
    G=nx.from_numpy_matrix(adj_mat)
    bge_scores[ind] = bge.score(G)

# save the results
np.save(score_path, bge_scores)