#are the 50 families structurally identical or do they vary?
# i got some trivially bs result but that was expected
# all families ARE different as per what nx.is_isomorphic() says.

import networkx as nx  # pyright: ignore[reportMissingModuleSource]
from collections import Counter, defaultdict
import sys
import os
from src.task1.data_loader import MetaFAMLoader
from src.task1.feature_extractor import RawFeatureExtractor
from src.task1.constants import PARENT_RELATIONS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def family_deets(G_fam, fam_nodes, features):
    
    deg_seq = tuple(sorted([G_fam.degree(n) for n in fam_nodes], reverse=True))
    gen_dist = tuple(sorted(Counter(
        features['people'][p]['generation'] for p in fam_nodes
        if features['people'][p]['generation'] is not None
    ).items()))
    
    return {
        'size': len(fam_nodes),
        'edges': G_fam.number_of_edges(),
        'density': round(nx.density(G_fam), 4),
        'degree_sequence': deg_seq,
        'generation_distribution': gen_dist,
        'diameter': nx.diameter(G_fam),
        'avg_clustering': round(nx.average_clustering(G_fam), 4),
    }



# CLAUDE GENERATED CODE BEGINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def compare_families(data_path='data/train.txt'):

    loader = MetaFAMLoader(data_path)
    loader.load()

    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()

    G = nx.Graph()
    for h, r, t in loader.triplets:
        G.add_edge(h, t)

    families = list(nx.connected_components(G))
    families.sort(key=lambda c: min(c))

    print(f"cross family comparisonn")

    fingerprints = []
    for i, fam_nodes in enumerate(families):
        G_fam = G.subgraph(fam_nodes)
        fp = family_deets(G_fam, fam_nodes, features)
        fp['index'] = i
        fingerprints.append(fp)

    # check degree sequence uniqueness
    deg_seqs = [fp['degree_sequence'] for fp in fingerprints]
    unique_deg = len(set(deg_seqs))
    print(f"Unique degree sequences: {unique_deg} / {len(families)}")

    # check generation distribution uniqueness
    gen_dists = [fp['generation_distribution'] for fp in fingerprints]
    unique_gen = len(set(gen_dists))
    print(f"Unique generation distributions: {unique_gen} / {len(families)}")

    # size variation
    sizes = [fp['size'] for fp in fingerprints]
    print(f"Size range: {min(sizes)}-{max(sizes)}, unique sizes: {Counter(sizes)}")

    # edge count variation
    edges = [fp['edges'] for fp in fingerprints]
    print(f"Edge count range: {min(edges)}-{max(edges)}, unique: {len(set(edges))}")

    # diameter
    diameters = [fp['diameter'] for fp in fingerprints]
    print(f"Diameter range: {min(diameters)}-{max(diameters)}, distribution: {Counter(diameters)}")

    # clustering
    clusterings = [fp['avg_clustering'] for fp in fingerprints]
    print(f"Clustering range: {min(clusterings)}-{max(clusterings)}")

    # graph isomorphism check on a sample
    print(f"\nPairwise isomorphism check (first 5 families):")

 
    for i in range(min(5, len(families))):
   
        for j in range(i+1, min(5, len(families))):
   
            Gi = G.subgraph(families[i])
            Gj = G.subgraph(families[j])
   
            iso = nx.is_isomorphic(Gi, Gj)          
            # main check that i wanted to do, but it says all families are different, 
            # which is surprising but degree seq and gen dist checks show some repetition but not a lot.
   
            print(f"  family {i} vs {j}: {'isomorphic' if iso else 'NOT isomorphic'}")

    if unique_deg == 1:
        print("  All families have identical degree sequences. structurally indistinguishable")

    elif unique_deg < len(families) // 2:

        print(f"  Only {unique_deg} distinct structures among {len(families)} families. high homogeneity")

    else:

        print(f"  {unique_deg} distinct structures. structural variation exists")

    return fingerprints


# CLAUDE GENERATED CODE ENDS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
    compare_families(data_path)