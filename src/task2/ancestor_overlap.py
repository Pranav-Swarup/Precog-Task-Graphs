# same metric i did in task 1 
# relatedness(A, B) = |ancestors(A) ∩ ancestors(B)| / |ancestors(A) ∪ ancestors(B)|

# apparently theres a word for it -> Jaccard ≈ 1.0
# if you have cousins and u share grandparents then jaccard is lower
# uses FamilyCentrality from tertiary_analysis.py

import networkx as nx # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore
from collections import defaultdict
from itertools import combinations
import sys
import os
from src.task1.data_loader import MetaFAMLoader
from src.task1.feature_extractor import RawFeatureExtractor
from src.task1.constants import PARENT_RELATIONS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_ancestor_sets(triplets, people):

    dag = nx.DiGraph()
    for h, r, t in triplets:
        if r in PARENT_RELATIONS:
            dag.add_edge(h, t)  # parent -> child
    
    reverse_dag = dag.reverse()
    
    ancestor_sets = {}
    for person in people:
        if person in reverse_dag:
            ancestor_sets[person] = set(nx.ancestors(reverse_dag, person))
        else:
            ancestor_sets[person] = set()
    
    return ancestor_sets


def ancestor_jaccard(ancestors_a, ancestors_b):

    if not ancestors_a and not ancestors_b:
        return 1.0  # both founders with no ancestors — equally rootless

    union = ancestors_a | ancestors_b
    if not union:
        return 0.0

    return len(ancestors_a & ancestors_b) / len(union)


def pairwise_ancestor_overlap(ancestor_sets, nodes):

    overlaps = {}
    for a, b in combinations(nodes, 2):
        pair = tuple(sorted([a, b]))
        overlaps[pair] = ancestor_jaccard(ancestor_sets[a], ancestor_sets[b])
    return overlaps


def rank_relatives_by_overlap(ancestor_sets, target_guy):

    target_guy_ancestors = ancestor_sets[target_guy]
    scores = []
    for person, anc in ancestor_sets.items():
        if person == target_guy:
            continue

        j = ancestor_jaccard(target_guy_ancestors, anc)
        scores.append((person, j))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def analyze_ancestor_overlap(triplets, features, sample_target_guys=None):
    
    people = list(features['people'].keys())
    ancestor_sets = build_ancestor_sets(triplets, people)
    
    G = nx.Graph()

    for h, r, t in triplets:
        G.add_edge(h, t)
    families = list(nx.connected_components(G))
    
    if sample_target_guys is None:
        fam0 = families[0]
        sample_target_guys = []

        for p in fam0:

            if features['people'][p]['generation'] == 2:
                sample_target_guys.append(p)
                break
    
    # build relation lookup for context
    direct_relations = defaultdict(dict)
    for h, r, t in triplets:
        direct_relations[h][t] = r
    
    print("ANCESTOR OVERLAP RELATEDNESS")
    
# CLAUDE GENERATED CODE BEGINS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    for target_guy in sample_target_guys:
        target_guy_gen = features['people'][target_guy]['generation']
        target_guy_anc = ancestor_sets[target_guy]
        print(f"\ntarget_guy: {target_guy} (gen {target_guy_gen}, {len(target_guy_anc)} ancestors)")
        
    # aggregate: average overlap by relation type
    print("AVERAGE ANCESTOR OVERLAP BY RELATION TYPE")
    
    rel_overlaps = defaultdict(list)
    for h, r, t in triplets:
        if h in ancestor_sets and t in ancestor_sets:
            j = ancestor_jaccard(ancestor_sets[h], ancestor_sets[t])
            rel_overlaps[r].append(j)
    
    sorted_rels = sorted(rel_overlaps.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"  {'Relation':<35} {'Mean':<8} {'Std':<8} {'N'}")
    for rel, scores in sorted_rels:
        print(f"  {rel:<35} {np.mean(scores):<8.4f} {np.std(scores):<8.4f} {len(scores)}")
    
    return ancestor_sets

# CLAUDE GENERATED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
    
    loader = MetaFAMLoader(data_path)
    loader.load()
    
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    analyze_ancestor_overlap(loader.triplets, features)