# calculate the transitive closure and Markov equivalence class for all the DAGs

import numpy as np
from tqdm import tqdm
import networkx as nx
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

n = 5
folder = '/data/zj448/causal/exact_posteriors'
DAGs = np.load(f'{folder}/dags_{n}.npy')
cores = 128
num_dags = len(DAGs)

def decompress_dag(dag, num_nodes):
    # decompress
    uncompressed=np.unpackbits(dag, axis=1, count=num_nodes**2).reshape(-1, num_nodes, num_nodes)
    return uncompressed

def cal_TC(dag):
    # Convert the DAG to a networkx graph
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    # Compute the transitive closure of the graph
    TC = nx.transitive_closure(G)
    # Convert the transitive closure graph back to a numpy array
    tc_adj_matrix = nx.to_numpy_array(TC, dtype=int)
    return tc_adj_matrix





def adj2DAG(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    cg = CausalGraph(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge1 = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i,j] == 1:
                cg.G.add_edge(Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))

    return cg.G

def cal_MEC(adj):
    # adjaceny matrix to DAG
    dag = adj2DAG(adj)
    # DAG to MEC/CPDAG
    cpdag = dag2cpdag(dag)
    MEC = cpdag.graph
    return MEC



DAGs = decompress_dag(DAGs, n)
Transitive_Closures = np.zeros((num_dags, n, n), dtype=np.int8)
Markov_Equivalence_Classes = np.zeros((num_dags, n, n), dtype=np.int8)
for i in tqdm(range(num_dags)):
    dag = DAGs[i]

    TC = cal_TC(dag)

    MEC = cal_MEC(dag)

    Transitive_Closures[i] = TC
    Markov_Equivalence_Classes[i] = MEC


# compress
Transitive_Closures = Transitive_Closures.reshape(-1,n**2)
#Markov_Equivalence_Classes = Markov_Equivalence_Classes.reshape(-1,n**2)

Transitive_Closures = np.packbits(Transitive_Closures, axis=1)
#Markov_Equivalence_Classes = np.packbits(Markov_Equivalence_Classes, axis=1)

# save the results
np.save(f'{folder}/TC_{n}.npy', Transitive_Closures)
np.save(f'{folder}/MEC_{n}.npy', Markov_Equivalence_Classes)