# this file is intentionally kept graph agnostic. 
# runs on any graph u throw at it.

import networkx as nx  # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
from collections import defaultdict
from community import community_louvain # type: ignore
from sklearn.cluster import SpectralClustering, KMeans # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from node2vec import Node2Vec   # pyright: ignore[reportMissingImports, reportMissingModuleSource]

def louvain(G, resolution=1.0, seed=42):

    return community_louvain.best_partition(G, resolution=resolution, random_state=seed)


def girvan_newman(G, k=None):

    comp = nx.community.girvan_newman(G)
    best_partition = None
    best_mod = -1

    for i, communities in enumerate(comp):

        if k is not None and len(communities) == k:
            return _communities_to_dict(communities)

        mod = nx.community.modularity(G, communities)

        if mod > best_mod:

            best_mod = mod
            best_partition = communities

        if k is None and i > 50:
            break

    return _communities_to_dict(best_partition)


# CLAUDE GENERATED CODE BEGINS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def spectral(G, n_clusters=None, seed=42):

    A = nx.to_numpy_array(G)
    nodes = list(G.nodes())

    if n_clusters is None:

        best_k, best_mod = 2, -1

        for k in range(2, min(len(nodes), 10)):

            labels = SpectralClustering(
                n_clusters=k, affinity='precomputed',
                random_state=seed, assign_labels='kmeans'
            ).fit_predict(A)

            comms = _labels_to_communities(nodes, labels)
            mod = nx.community.modularity(G, comms)

            if mod > best_mod:
                best_mod = mod
                best_k = k

        n_clusters = best_k

    labels = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        random_state=seed, assign_labels='kmeans'
    ).fit_predict(A)

    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}

def node2vec_kmeans(G, n_clusters=None, dimensions=32, walk_length=10,
                    num_walks=80, seed=42):


    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                   num_walks=num_walks, seed=seed, quiet=True)

    model = n2v.fit(window=5, min_count=1, seed=seed)

    nodes = list(G.nodes())
    embeddings = np.array([model.wv[str(n)] for n in nodes])

    if n_clusters is None:
        best_k, best_mod = 2, -1
        for k in range(2, min(len(nodes), 10)):
            labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
            comms = _labels_to_communities(nodes, labels)
            mod = nx.community.modularity(G, comms)
            if mod > best_mod:
                best_mod = mod
                best_k = k
        n_clusters = best_k

    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embeddings)
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}


# CLAUDE GENERATED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def label_propagation(G):
    communities = nx.community.label_propagation_communities(G)
    return _communities_to_dict(communities)



def evaluate(G, partition):

    comms = defaultdict(set)

    for node, cid in partition.items():
        comms[cid].add(node)

    community_sets = list(comms.values())

    mod = nx.community.modularity(G, community_sets)
    sizes = [len(c) for c in community_sets]

    return {
        'modularity': mod,
        'n_communities': len(community_sets),
        'sizes': sorted(sizes, reverse=True),
        'avg_size': np.mean(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
    }


def _communities_to_dict(communities):
    result = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            result[node] = cid
    return result


def _labels_to_communities(nodes, labels):
    comms = defaultdict(set)
    for node, label in zip(nodes, labels):
        comms[label].add(node)
    return list(comms.values())