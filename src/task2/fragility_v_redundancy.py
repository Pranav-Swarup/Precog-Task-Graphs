# Generational Fragility: remove each person, measure what breaks.

import networkx as nx # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task1.data_loader import MetaFAMLoader
from src.task1.feature_extractor import RawFeatureExtractor
from src.task2.graph_builder import *

def removal_impact(G, node):

    if node not in G:
        return None

    original_components = nx.number_connected_components(G)
    original_pairs = sum(len(c) * (len(c) - 1) // 2 for c in nx.connected_components(G))

    G_removed = G.copy()
    G_removed.remove_node(node)

    new_components = nx.number_connected_components(G_removed)
    new_pairs = sum(len(c) * (len(c) - 1) // 2 for c in nx.connected_components(G_removed))

    return {
        'component_increase': new_components - original_components,
        'reachability_loss': original_pairs - new_pairs,
        'reachability_loss_frac': 1 - (new_pairs / original_pairs) if original_pairs > 0 else 0,
        'fragments': sorted([len(c) for c in nx.connected_components(G_removed)], reverse=True),
    }


def dag_removal_impact(dag, node):
    
    if node not in dag:
        return {'orphaned_descendants': 0, 'n_descendants': 0, 'n_ancestors': 0}

    descendants = nx.descendants(dag, node)
    ancestors = nx.ancestors(dag, node)

    dag_removed = dag.copy()
    dag_removed.remove_node(node)

    orphaned = 0
    for desc in descendants:
        still_connected = False
        for anc in ancestors:
            if anc in dag_removed and desc in dag_removed:
                if nx.has_path(dag_removed, anc, desc):
                    still_connected = True
                    break
        if not still_connected:
            orphaned += 1

    return {
        'orphaned_descendants': orphaned,
        'n_descendants': len(descendants),
        'n_ancestors': len(ancestors),
    }


def generational_disconnection(G, node, generation_map):

    if node not in G:
        return {'generations_split': False}

    G_removed = G.copy()
    G_removed.remove_node(node)

    components = list(nx.connected_components(G_removed))
    if len(components) <= 1:
        return {'generations_split': False, 'n_fragments': 1, 'isolated_generations': []}

    gen_sets = []
    for comp in components:
        gens = {generation_map[n] for n in comp if n in generation_map and generation_map[n] is not None}
        gen_sets.append(gens)

    all_gens = set().union(*gen_sets)
    isolated_gens = [g for g in all_gens if sum(1 for gs in gen_sets if g in gs) == 1]

    return {
        'generations_split': True,
        'n_fragments': len(components),
        'isolated_generations': isolated_gens,
    }


def analyze_family(G_undirected, dag, family_nodes, generation_map):
    
    results = []
    for node in family_nodes:
        gen = generation_map.get(node, -1)
        undirected = removal_impact(G_undirected, node)
        directed = dag_removal_impact(dag, node)
        gen_disc = generational_disconnection(G_undirected, node, generation_map)

        results.append({
            'node': node,
            'generation': gen,
            **(undirected or {}),
            **directed,
            **gen_disc,
        })
    return results


def summarize_by_generation(results):
    
    by_gen = defaultdict(list)
    for r in results:
        by_gen[r['generation']].append(r)

    print(f"{'Gen':<5} {'Count':<6} {'Avg Comp+':<10} {'Avg Reach%':<12} {'Avg Orphan':<12} {'Splits?':<8}")
    print("-" * 55)
    for gen in sorted(by_gen.keys()):
        entries = by_gen[gen]
        avg_comp = np.mean([e.get('component_increase', 0) for e in entries])
        avg_reach = np.mean([e.get('reachability_loss_frac', 0) for e in entries]) * 100
        avg_orphan = np.mean([e.get('orphaned_descendants', 0) for e in entries])
        any_split = any(e.get('generations_split', False) for e in entries)
        print(f"{gen:<5} {len(entries):<6} {avg_comp:<10.2f} {avg_reach:<12.2f} {avg_orphan:<12.2f} {str(any_split):<8}")


if __name__ == '__main__':

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'

    loader = MetaFAMLoader(data_path)
    loader.load()

    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    
    features = extractor.extract_all()

    generation_map = {pid: f['generation'] for pid, f in features['people'].items()}

    G = full_undirected(loader.triplets)
    
    dag = parent_child_dag(loader.triplets)
    families = list(get_families(G))

    print(f"Analyzing {len(families)} families\n")

    for i, fam_nodes in enumerate(families[:5]):
    
        G_fam = family_subgraph(G, fam_nodes)
        dag_fam = dag.subgraph(fam_nodes).copy()

        print(f"\n{'='*50}")
        print(f"FAMILY {i} ({len(fam_nodes)} members)")
        print(f"{'='*50}")

        results = analyze_family(G_fam, dag_fam, fam_nodes, generation_map)
        summarize_by_generation(results)

        ranked = sorted(results, key=lambda r: r.get('reachability_loss_frac', 0), reverse=True)
        print(f"\nMost disruptive removals:")
        for r in ranked[:5]:
            print(f"  {r['node']} (gen {r['generation']}): "
                  f"reach_loss={r.get('reachability_loss_frac', 0)*100:.1f}%, "
                  f"orphaned={r.get('orphaned_descendants', 0)}, "
                  f"fragments={r.get('fragments', [])}")