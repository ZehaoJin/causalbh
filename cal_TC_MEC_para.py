import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from tqdm import tqdm
from math import ceil

n = 7
folder = '/data/zj448/causal/exact_posteriors'
DAGs = np.load(f'{folder}/dags_{n}.npy')
num_dags = len(DAGs)
cores = 128  # Adjust this number based on your machine's CPU cores


def decompress_dag(dag, num_nodes=n):
    #print(dag.shape)
    uncompressed = np.unpackbits(dag, axis=1, count=num_nodes**2).reshape(-1, num_nodes, num_nodes)
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




# parallel processing
def process_chunk(chunk):
    chunk = decompress_dag(chunk, n)
    TC_results = np.zeros((len(chunk), n, n), dtype=np.int8)
    MEC_results = np.zeros((len(chunk), n, n), dtype=np.int8)
    for i in range(len(chunk)):
        TC = cal_TC(chunk[i])
        MEC = cal_MEC(chunk[i])

        TC_results[i] = TC
        MEC_results[i] = MEC

    # Compress results
    TC_results = np.packbits(TC_results.reshape(-1, n**2), axis=1)
    # can't compress MEC for now
    # MEC_results = np.packbits(MEC_results.reshape(-1, n**2), axis=1)
        
    
    return TC_results, MEC_results

# Split DAGs into approximately equal chunks for each core
def split_into_chunks(lst, num_chunks):
    """
    Splits the list into num_chunks chunks, with the last chunk being equal or larger than the others.
    """
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []

    start = 0
    for i in range(num_chunks):
        # If we're at the last chunk, include the remainder
        if i == num_chunks - 1:
            end = None
        else:
            end = start + chunk_size + (1 if remainder > 0 else 0)
            remainder -= 1

        chunks.append(lst[start:end])
        start = end

    return chunks




chunks = split_into_chunks(DAGs, cores)

with ProcessPoolExecutor(max_workers=cores) as executor:
    futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
    
    # Setup progress bar for the number of chunks
    with tqdm(total=len(futures), desc="Processing Chunks") as progress:
        for future in as_completed(futures):
            # Result retrieval
            result = future.result()
            
            # Update progress bar upon task completion
            progress.update(1)
    
    # Collecting results might need adjustment depending on how you handle them
    results = [future.result() for future in futures]

# Combine the results
TC_compressed_results = np.concatenate([result[0] for result in results])
MEC_compressed_results = np.concatenate([result[1] for result in results])

# Save the compressed results
np.save(f'{folder}/TC_{n}.npy', TC_compressed_results)
np.save(f'{folder}/MEC_{n}.npy', MEC_compressed_results)