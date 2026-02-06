# I would've put hagrid's relation edge weight higher than harry's own parents

import networkx as nx # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task1.data_loader import MetaFAMLoader
from src.task1.feature_extractor import RawFeatureExtractor
from src.task2.graph_builder import RELATION_DISTANCE

def build_distance_graph(triplets):

    G = nx.Graph()
    for h, r, t in triplets:
        dist = RELATION_DISTANCE.get(r, 2.0)
        if G.has_edge(h, t):
            G[h][t]['weight'] = min(G[h][t]['weight'], dist)
        else:
            G.add_edge(h, t, weight=dist)
    return G


def pairwise_weighted_distance(G, nodes=None):
    
    #dijkstra but for all nodes

    if nodes is None:
        nodes = list(G.nodes())
    
    distances = {}
    for source in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
        for target in nodes:
            if target != source:
                pair = tuple(sorted([source, target]))
                if pair not in distances:
                    distances[pair] = lengths.get(target, float('inf'))
    return distances


def rank_relatives(G, targetguy, features=None):

    lengths = nx.single_source_dijkstra_path_length(G, targetguy, weight='weight')
    ranked = sorted(
        [(person, dist) for person, dist in lengths.items() if person != targetguy],
        key=lambda x: x[1]
    )
    
    if features:
        ranked = [(p, d, features['people'][p]['generation']) for p, d in ranked]
    
    return ranked


def analyze_distance_stratification(triplets, features, sample_targetguys=None):
    
    G_weighted = build_distance_graph(triplets)
    
    G_unweighted = nx.Graph()
    for h, r, t in triplets:
        G_unweighted.add_edge(h, t)
    
    families = list(nx.connected_components(G_unweighted))
    
    if sample_targetguys is None:

        # pick one targetguy from first family, gen 2 (middle generation, interesting connections)
        
        fam0 = families[0]
        sample_targetguys = []
        
        for p in fam0:
        
            if features['people'][p]['generation'] == 2:
                sample_targetguys.append(p)
                break
    
    print("relation weighed distance analysis")
    
    for targetguy in sample_targetguys:

        print(f"\ntargetguy: {targetguy} (gen {features['people'][targetguy]['generation']})")
        
        
        hop_lengths = dict(nx.single_source_shortest_path_length(G_unweighted, targetguy))
        
        
        weighted_lengths = dict(nx.single_source_dijkstra_path_length(G_weighted, targetguy, weight='weight'))

# CLAUDE GENERATED CODE BEGINS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        by_hops = defaultdict(list)
        for person in hop_lengths:
            if person == targetguy:
                continue
            h = hop_lengths[person]
            w = weighted_lengths.get(person, float('inf'))
            gen = features['people'][person]['generation']
            by_hops[h].append((person, w, gen))
        
        print(f"\n{'Hops':<6} {'Count':<7} {'Weighted Range':<20} {'Example'}")

        for h in sorted(by_hops.keys()):
            entries = by_hops[h]
            ws = [e[1] for e in entries]
            # pick example with min and max weighted distance
            entries_sorted = sorted(entries, key=lambda x: x[1])
            closest = entries_sorted[0]
            farthest = entries_sorted[-1]
            print(f"{h:<6} {len(entries):<7} [{min(ws):.1f} - {max(ws):.1f}]"
                  f"{'':>5} closest={closest[0]}({closest[1]:.1f}), "
                  f"farthest={farthest[0]}({farthest[1]:.1f})")
        
        # full ranking
        ranked = rank_relatives(G_weighted, targetguy, features)
        print(f"\nTop 10 closest by weighted distance:")
        for person, dist, gen in ranked[:20]:
            hops = hop_lengths.get(person, '?')
            # find the direct relation if it exists
            direct_rel = None
            for h, r, t in triplets:
                if (h == targetguy and t == person) or (h == person and t == targetguy):
                    direct_rel = r
                    break
            rel_str = f" [{direct_rel}]" if direct_rel else ""
            print(f"  {person:<15} dist={dist:.2f}  hops={hops}  gen={gen}{rel_str}")
    
    return G_weighted


# CLAUDE GENERATED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
    
    loader = MetaFAMLoader(data_path)
    loader.load()
    
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    analyze_distance_stratification(loader.triplets, features)