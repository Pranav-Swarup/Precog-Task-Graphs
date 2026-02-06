# this script runs the various techniques from community_detection.py and prints the solution.
# Hence LLM was used to generate this auto running script.

# CLAUDE GENERATED CODE STARTS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sys
import os
import numpy as np  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task1.data_loader import MetaFAMLoader
from src.task2.graph_builder import *
from src.task2.community_detection import *

ALGORITHMS = {
    'louvain': louvain,
    'girvan_newman': girvan_newman,
    'spectral': spectral,
    'label_propagation': label_propagation,
    'node2vec_kmeans': node2vec_kmeans,
}

GRAPH_BUILDERS = {
    'full_undirected': full_undirected,
    'nuclear_family': nuclear_family_graph,
    'generational_affinity': generational_affinity_graph,
    'relation_weighted': relation_weighted_graph,
}


def run_on_graph(G, algo_name, algo_fn):
    if G.number_of_nodes() < 3:
        return None, None
    try:
        partition = algo_fn(G)
        metrics = evaluate(G, partition)
        return partition, metrics
    except Exception as e:
        print(f"  {algo_name} failed: {e}")
        return None, None


def run_full_graph(triplets):

    print("FULL GRAPH")

    G = full_undirected(triplets)
    
    results = {}
    for name, fn in ALGORITHMS.items():
        print(f"Running {name}...")
        partition, metrics = run_on_graph(G, name, fn)
        if metrics:
            results[name] = metrics
            print(f"  communities: {metrics['n_communities']}, "
                  f"modularity: {metrics['modularity']:.4f}")
            print(f"  sizes: {metrics['sizes'][:10]}{'...' if len(metrics['sizes']) > 10 else ''}")
        print()

    return results


def run_intra_family(triplets, graph_type='full_undirected', max_families=None):

    print(f"INTRA-FAMILY | {graph_type} graph")
    
    G_full = full_undirected(triplets)
    families = list(get_families(G_full))
    builder = GRAPH_BUILDERS[graph_type]

    if max_families:
        families = families[:max_families]

    algo_results = {name: [] for name in ALGORITHMS}

    for i, fam_nodes in enumerate(families):
        fam_trips = family_triplets(triplets, fam_nodes)
        G_fam = builder(fam_trips)

        for node in fam_nodes:
            if node not in G_fam:
                G_fam.add_node(node)

        if G_fam.number_of_edges() < 2:
            continue

        for name, fn in ALGORITHMS.items():
            partition, metrics = run_on_graph(G_fam, name, fn)
            if metrics:
                algo_results[name].append(metrics)

    print()
    for name, results_list in algo_results.items():
        if not results_list:
            print(f"{name}: no results")
            continue

        mods = [r['modularity'] for r in results_list]
        n_comms = [r['n_communities'] for r in results_list]
        print(f"{name} (across {len(results_list)} families):")
        print(f"  modularity: mean={np.mean(mods):.4f}, std={np.std(mods):.4f}, "
              f"range=[{min(mods):.4f}, {max(mods):.4f}]")
        print(f"  communities: mean={np.mean(n_comms):.1f}, "
              f"range=[{min(n_comms)}, {max(n_comms)}]")
        print()

    return algo_results


def compare_graph_representations(triplets, family_index=0):
    
    
    print(f"GRAPH REPRESENTATION COMPARISON (family {family_index})")

    G_full = full_undirected(triplets)
    families = list(get_families(G_full))
    fam_nodes = families[family_index]
    fam_trips = family_triplets(triplets, fam_nodes)

    print(f"Family size: {len(fam_nodes)} nodes\n")

    for gtype, builder in GRAPH_BUILDERS.items():
        G = builder(fam_trips)
        for node in fam_nodes:
            if node not in G:
                G.add_node(node)

        print(f"--- {gtype} ({G.number_of_edges()} edges) ---")
        for aname, afn in ALGORITHMS.items():
            partition, metrics = run_on_graph(G, aname, afn)
            if metrics:
                print(f"  {aname}: {metrics['n_communities']} comms, "
                      f"mod={metrics['modularity']:.4f}, sizes={metrics['sizes']}")
        print()


if __name__ == '__main__':

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'

    loader = MetaFAMLoader(data_path)
    loader.load()
    triplets = loader.triplets
    print(f"Loaded {len(triplets)} triplets\n")

    run_full_graph(triplets)
    run_intra_family(triplets, graph_type='full_undirected', max_families=10)
    compare_graph_representations(triplets, family_index=0)

    print("\n\nNuclear family graph representation\n")
    run_intra_family(triplets, graph_type='nuclear_family', max_families=10)

# CLAUDE GENERATED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!