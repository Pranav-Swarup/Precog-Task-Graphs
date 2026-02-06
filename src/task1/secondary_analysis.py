# computes standard graph theory metrics + family-specific stuff

import networkx as nx
from collections import defaultdict, Counter
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task1.data_loader import MetaFAMLoader
from src.task1.feature_extractor import RawFeatureExtractor
from src.task1.inference import infer_gender
from src.task1.constants import *


class GraphMetrics:
    
    def __init__(self, triplets, features):
        self.triplets = triplets
        self.features = features
        self.people = list(features['people'].keys())
        
        # with digraphs we lose edge multiplicity info 
        # but some algorithms work here which dont with multidigraphs

        self.G_directed = nx.DiGraph()
        for h, r, t in triplets:
            self.G_directed.add_edge(h, t, relation=r)
        
        
        self.G_undirected = self.G_directed.to_undirected()
        
        # indexes
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for h, r, t in triplets:
            self.outgoing[h].append((r, t))
            self.incoming[t].append((r, h))
    
    
    
    def basic_stats(self):

        n_nodes = self.G_undirected.number_of_nodes()
        n_edges = self.G_undirected.number_of_edges()
        
        # density = actual edges / possible edges
        # possible edges in undirected = n*(n-1)/
        
        density = nx.density(self.G_undirected)
        
        # avg degree
        degrees = [d for n, d in self.G_undirected.degree()]
        avg_degree = sum(degrees) / len(degrees)
        
        
        print(f"nodes: {n_nodes}")
        print(f"edges (undirected): {n_edges}")

        print(f"edges (directed/triplets): {len(self.triplets)}")

        print(f"density: {density:.6f}")
        
        print(f"avg degree: {avg_degree:.2f}")
        print(f"degree range: {min(degrees)} - {max(degrees)}")
        
        return {
            'n_nodes': n_nodes,
            'n_edges_undirected': n_edges,
            'n_triplets': len(self.triplets),
            'density': density,
            
            'avg_degree': avg_degree,
            'min_degree': min(degrees),
            'max_degree': max(degrees),
        }
    
    
    def connected_components(self):
    
        components = list(nx.connected_components(self.G_undirected))

        sizes = sorted([len(c) for c in components], reverse=True)
        
        
        print(f"number of components: {len(components)}")
        
        print(f"  each family has {sizes[0]}-{sizes[-1] if len(sizes) > 1 else sizes[0]} members")

        print(f"largest component: {sizes[0]} nodes ({sizes[0]/len(self.people)*100:.1f}% of graph)")
        
        print(f"singleton components: {sum(1 for s in sizes if s == 1)}")
        
        print(f"size distribution: {Counter(sizes)}")
        
        return {
            'n_components': len(components),
            'largest_size': sizes[0],
            'size_distribution': dict(Counter(sizes)),
            'components': components,  # keep for later analysis
        }
    
    
    def diameter_and_paths(self):

        # get largest component
        largest_cc = max(nx.connected_components(self.G_undirected), key=len)
        G_largest = self.G_undirected.subgraph(largest_cc)
        
        n = len(largest_cc)
        
        # the family sizes here are pretty small enough for exact calculation
        diameter = nx.diameter(G_largest)
        avg_path = nx.average_shortest_path_length(G_largest)
        
        print(f"diameter: {diameter}")
        print(f"avg shortest path length: {avg_path:.2f}")
        
        
        result = {
            'exact': True,
            'diameter': diameter,
            'avg_path_length': avg_path,
        }
        
        return result
    
    
    def clustering_analysis(self):


        # global clustering
        avg_clustering = nx.average_clustering(self.G_undirected)
        
        # per-node clustering
        clustering = nx.clustering(self.G_undirected)
        
        # find nodes with highest/lowest clustering
        sorted_by_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
        
        # correlate with generation
        gen_clustering = defaultdict(list)
        for person, cc in clustering.items():
            gen = self.features['people'][person]['generation']
            if gen is not None:
                gen_clustering[gen].append(cc)
        
        gen_avg_clustering = {g: sum(vals)/len(vals) for g, vals in gen_clustering.items() if vals}
        
        
        print(f"global avg clustering: {avg_clustering:.4f}")
        
        print(f"\nhighest clustering (most interconnected neighbors):")
        
        for person, cc in sorted_by_clustering[:5]:
            gen = self.features['people'][person]['generation']
            print(f"  {person}: {cc:.3f} (gen {gen})")
        
        
        print(f"\nlowest clustering (potential bridges):")
        
        for person, cc in sorted_by_clustering[-5:]:
            gen = self.features['people'][person]['generation']
            print(f"  {person}: {cc:.3f} (gen {gen})")
        
        
        print(f"\navg clustering by generation:")
        
        for g in sorted(gen_avg_clustering.keys()):
            print(f"  gen {g}: {gen_avg_clustering[g]:.4f}")
        
        return {
            'global_avg': avg_clustering,
            'per_node': clustering,
            'by_generation': gen_avg_clustering,
            'highest': sorted_by_clustering[:10],
            'lowest': sorted_by_clustering[-10:],
        }
    
    
    
    def centrality_analysis(self):
        
        components = list(nx.connected_components(self.G_undirected))
        
        
        # here im using largest component for meaningful results
        largest_cc = max(components, key=len)
        G = self.G_undirected.subgraph(largest_cc)
        
        print(f"centrality analysis of component with {len(largest_cc)} members")
        
        
        degree_cent = nx.degree_centrality(G)
        
        betweenness = nx.betweenness_centrality(G)
        
        closeness = nx.closeness_centrality(G)
 

# CLAUDE ASSISTED CODE STARTS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        
        except nx.PowerIterationFailedConvergence:
            print("  WARNING: eigenvector centrality didn't converge, using pagerank instead")
            eigenvector = nx.pagerank(G)



# CLAUDE ASSISTED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        # find top nodes by each measure
        def top_n(cent_dict, n=10):
            return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:n]


        print("\nDEGREE CENTRALITY (most connections):")
        for person, score in top_n(degree_cent, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nBETWEENNESS CENTRALITY (bridge nodes):")
        for person, score in top_n(betweenness, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nCLOSENESS CENTRALITY (central position):")
        for person, score in top_n(closeness, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nEIGENVECTOR/PAGERANK CENTRALITY (influence):")
        
        for person, score in top_n(eigenvector, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        # analyze correlation between centralities
        # high betweenness + low degree = true bridge
        # high everything = central family figure
        
        bridge_candidates = []
        central_figures = []
        
        for person in G.nodes():
            bc = betweenness.get(person, 0)
            dc = degree_cent.get(person, 0)
            
            # bridge: high betweenness relative to degree
            if bc > 0.01 and dc < 0.05:
                bridge_candidates.append((person, bc, dc))
            
            # central: high on everything
            if bc > 0.01 and dc > 0.03 and closeness.get(person, 0) > 0.3:
                central_figures.append((person, bc, dc, closeness[person]))
        
        print(f"\nBRIDGE CANDIDATES (high betweenness, low degree):")
        
        for person, bc, dc in sorted(bridge_candidates, key=lambda x: x[1], reverse=True)[:5]:
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: betweenness={bc:.4f}, degree_cent={dc:.4f} (gen {gen})")
        
        print(f"\nCENTRAL FAMILY FIGURES (high on all metrics):")
        for person, bc, dc, cc in sorted(central_figures, key=lambda x: x[1], reverse=True)[:5]:
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: betweenness={bc:.4f}, degree={dc:.4f}, closeness={cc:.4f} (gen {gen})")
        
        return {
            'degree': degree_cent,
            'betweenness': betweenness,
            'closeness': closeness,
            'eigenvector': eigenvector,
            'bridge_candidates': bridge_candidates,
        }
    

    
    
    def generation_analysis(self):

        gen_stats = defaultdict(lambda: {
            'count': 0,
            'total_degree': 0,
            'founders': 0,  # no parents
            'leaves': 0,    # no children
        })
        
        for person, f in self.features['people'].items():
            gen = f['generation']
            if gen is None:
                continue
            
            gen_stats[gen]['count'] += 1
            gen_stats[gen]['total_degree'] += f['relation_counts']['total']
            
            if f['parents']['total'] == 0:
                gen_stats[gen]['founders'] += 1

            if f['children']['total'] == 0:
                gen_stats[gen]['leaves'] += 1
        
        print("\n--- generation analysis---")
        print(f"{'Gen':<5} {'Count':<8} {'Avg Deg':<10} {'Founders':<10} {'Leaves':<10}")
        print("-" * 45)
        
        for g in sorted(gen_stats.keys()):
            s = gen_stats[g]
            avg_deg = s['total_degree'] / s['count'] if s['count'] > 0 else 0
            print(f"{g:<5} {s['count']:<8} {avg_deg:<10.2f} {s['founders']:<10} {s['leaves']:<10}")
        
        return dict(gen_stats)
    
    # this is just a check here for DAG

    def relation_subgraph_analysis(self):
        
        # parent-child subgraph (should be DAG / forest)
        parent_child_edges = [(h, t) for h, r, t in self.triplets if r in PARENT_RELATIONS]
        G_pc = nx.DiGraph()
        G_pc.add_edges_from(parent_child_edges)
        
        is_dag = nx.is_directed_acyclic_graph(G_pc)
        pc_components = nx.number_weakly_connected_components(G_pc)

        if is_dag:
            print(f"  VALID: parent-child forms proper forest structure")
        else:
            print(f"  WARNING: cycles detected - possible data error")
        
        # sibling subgraph (should be cliques)
        sibling_edges = [(h, t) for h, r, t in self.triplets if r in SIBLING_RELATIONS]
        G_sib = nx.Graph()
        G_sib.add_edges_from(sibling_edges)
        
        sib_components = list(nx.connected_components(G_sib))
        
        # check if each component is a clique
        clique_violations = 0
        for comp in sib_components:
            subg = G_sib.subgraph(comp)
            n = len(comp)
            expected_edges = n * (n - 1) // 2
            actual_edges = subg.number_of_edges()
            if actual_edges < expected_edges:
                clique_violations += 1
        
        if clique_violations == 0:
            print(f"  VALID: all sibling groups fully connected")
        else:
            print(f"  NOTE: {clique_violations} groups have missing sibling edges")
            print(f"  this is expected if data doesn't encode all sibling pairs explicitly")
        
        return {
            'parent_child': {
                'edges': len(parent_child_edges),
                'is_dag': is_dag,
                'components': pc_components,
            },
            'sibling': {
                'edges': len(sibling_edges),
                'groups': len(sib_components),
                'incomplete_cliques': clique_violations,
            },
        }
    
# analysis since we have 50 families, to see if there are common patterns or differences between families


    def cross_family_analysis(self):

        components = list(nx.connected_components(self.G_undirected))
        
        print("\n=== CROSS-FAMILY AGGREGATE ANALYSIS ===")
        print(f"analyzing {len(components)} separate family units\n")
        
        family_stats = []
        
        for i, comp in enumerate(components):
            G_fam = self.G_undirected.subgraph(comp)
            
            # basic stats
            n = len(comp)
            m = G_fam.number_of_edges()
            density = nx.density(G_fam)
            avg_clustering = nx.average_clustering(G_fam)
            diameter = nx.diameter(G_fam)
            avg_path = nx.average_shortest_path_length(G_fam)
            
            # generation spread in this family
            gens = [self.features['people'][p]['generation'] for p in comp 
                    if self.features['people'][p]['generation'] is not None]
            gen_span = max(gens) - min(gens) if gens else 0
            
            family_stats.append({
                'size': n,
                'edges': m,
                'density': density,
                'clustering': avg_clustering,
                'diameter': diameter,
                'avg_path': avg_path,
                'gen_span': gen_span,
            })
        
        # aggregate stats
        print("FAMILY SIZE:")
        sizes = [f['size'] for f in family_stats]
        print(f"  range: {min(sizes)} - {max(sizes)}")
        print(f"  mean: {sum(sizes)/len(sizes):.1f}")
        
        print("\nDENSITY:")
        densities = [f['density'] for f in family_stats]
        print(f"  range: {min(densities):.4f} - {max(densities):.4f}")
        print(f"  mean: {sum(densities)/len(densities):.4f}")
        
        print("\nCLUSTERING COEFFICIENT:")
        clusterings = [f['clustering'] for f in family_stats]
        print(f"  range: {min(clusterings):.4f} - {max(clusterings):.4f}")
        print(f"  mean: {sum(clusterings)/len(clusterings):.4f}")
        print(f"  INSIGHT: consistently high clustering ({sum(clusterings)/len(clusterings):.2f}) confirms tight family structure")
        
        print("\nDIAMETER:")
        diameters = [f['diameter'] for f in family_stats]
        print(f"  range: {min(diameters)} - {max(diameters)}")
        print(f"  most common: {Counter(diameters).most_common(1)[0]}")
        print(f"  INSIGHT: small diameter ({Counter(diameters).most_common(1)[0][0]}) means everyone closely connected")
        
        print("\nAVG PATH LENGTH:")
        paths = [f['avg_path'] for f in family_stats]
        print(f"  range: {min(paths):.2f} - {max(paths):.2f}")
        print(f"  mean: {sum(paths)/len(paths):.2f}")
        
        print("\nGENERATION SPAN:")
        spans = [f['gen_span'] for f in family_stats]
        print(f"  range: {min(spans)} - {max(spans)}")
        print(f"  distribution: {Counter(spans)}")
        print(f"  INSIGHT: families span {min(spans)}-{max(spans)} generations")
        
        return family_stats

    

def run_all_metrics(data_path='data/train.txt'):
    
    # load
    print("\nLoading data...")
    loader = MetaFAMLoader(data_path)
    loader.load()
    
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    # run metrics
    metrics = GraphMetrics(loader.triplets, features)
    
    results = {}
    
    results['basic'] = metrics.basic_stats()
    results['components'] = metrics.connected_components()
    results['diameter'] = metrics.diameter_and_paths()
    results['clustering'] = metrics.clustering_analysis()
    results['centrality'] = metrics.centrality_analysis()
    
    results['generations'] = metrics.generation_analysis()
    results['relation_subgraphs'] = metrics.relation_subgraph_analysis()
    results['cross_family'] = metrics.cross_family_analysis()
    
    # save results
    return metrics, results


if __name__ == "__main__":
    random.seed(42)  # reproducibility for sampling
    metrics, results = run_all_metrics()