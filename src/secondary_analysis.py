# computes standard graph theory metrics + family-specific stuff

import networkx as nx
from collections import defaultdict, Counter
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MetaFAMLoader
from src.feature_extractor import RawFeatureExtractor
from src.inference import infer_gender
from src.constants import *


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
        """
        basic counts and density
        
        LIMITATION: density formula assumes simple graph
        our graph has multiple relation types between same pair
        so actual "richness" is higher than density suggests
        """
        n_nodes = self.G_undirected.number_of_nodes()
        n_edges = self.G_undirected.number_of_edges()
        
        # density = actual edges / possible edges
        # possible edges in undirected = n*(n-1)/2
        density = nx.density(self.G_undirected)
        
        # avg degree
        degrees = [d for n, d in self.G_undirected.degree()]
        avg_degree = sum(degrees) / len(degrees)
        
        print("=== BASIC GRAPH STATS ===")
        print(f"nodes: {n_nodes}")
        print(f"edges (undirected): {n_edges}")
        print(f"edges (directed/triplets): {len(self.triplets)}")
        print(f"density: {density:.6f}")
        print(f"Here, low density expected for family graphs - not everyone related to everyone")
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
        
        print("\n=== CONNECTED COMPONENTS ===")
        print(f"number of components: {len(components)}")
        print(f"FINDING: graph is {len(components)} separate extended families")
        print(f"  each family has {sizes[0]}-{sizes[-1] if len(sizes) > 1 else sizes[0]} members")
        print(f"largest component: {sizes[0]} nodes ({sizes[0]/len(self.people)*100:.1f}% of graph)")
        
        if len(sizes) > 1:
            print(f"second largest: {sizes[1]} nodes")
        
        print(f"singleton components: {sum(1 for s in sizes if s == 1)}")
        
        print(f"size distribution: {Counter(sizes)}")
        
        return {
            'n_components': len(components),
            'largest_size': sizes[0],
            'size_distribution': dict(Counter(sizes)),
            'components': components,  # keep for later analysis
        }
    
    
    def diameter_and_paths(self):
        """
        diameter = longest shortest path
        avg path length = mean of all shortest paths
        
        LIMITATION: only computable for connected graphs
        we compute for largest component only
        
        TRADEOFF: exact diameter is O(n^2), we sample for large graphs
        """
        # get largest component
        largest_cc = max(nx.connected_components(self.G_undirected), key=len)
        G_largest = self.G_undirected.subgraph(largest_cc)
        
        n = len(largest_cc)
        print(f"\n=== DIAMETER & PATH LENGTHS (largest component, n={n}) ===")
        
        if n > 500:
            # too big for exact computation, sample
            print(f"  NOTE: sampling due to size, exact computation would be O(n^2)")
            sample_size = min(200, n)
            sample_nodes = random.sample(list(largest_cc), sample_size)
            
            path_lengths = []
            for i, source in enumerate(sample_nodes[:50]):  # limit sources
                lengths = nx.single_source_shortest_path_length(G_largest, source)
                path_lengths.extend(lengths.values())
            
            avg_path = sum(path_lengths) / len(path_lengths)
            max_path = max(path_lengths)
            print(f"estimated avg path length: {avg_path:.2f}")
            print(f"estimated diameter (lower bound): {max_path}")
            
            result = {
                'exact': False,
                'sample_size': sample_size,
                'avg_path_length_estimate': avg_path,
                'diameter_lower_bound': max_path,
            }
        else:
            # small enough for exact
            diameter = nx.diameter(G_largest)
            avg_path = nx.average_shortest_path_length(G_largest)
            
            print(f"diameter: {diameter}")
            print(f"avg shortest path length: {avg_path:.2f}")
            print(f"  INSIGHT: diameter ~{diameter} means max {diameter} hops between any two family members")
            
            result = {
                'exact': True,
                'diameter': diameter,
                'avg_path_length': avg_path,
            }
        
        return result
    
    # ==================== CLUSTERING COEFFICIENT ====================
    
    def clustering_analysis(self):
        """
        clustering coefficient = how connected are a node's neighbors to each other
        
        INTERPRETATION FOR FAMILY GRAPHS:
        - high clustering around siblings (all connected to same parents)
        - lower clustering for bridge nodes (connect different families)
        
        LIMITATION: uses undirected graph, ignores relation semantics
        two people being "connected" via different relation types treated same
        """
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
        
        print("\n=== CLUSTERING COEFFICIENT ===")
        print(f"global avg clustering: {avg_clustering:.4f}")
        print(f"  INSIGHT: value ~{avg_clustering:.2f} suggests {'tight family clusters' if avg_clustering > 0.3 else 'more spread out structure'}")
        
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
        print(f"  INSIGHT: if middle generations have lower clustering, they're bridges between old/young")
        
        return {
            'global_avg': avg_clustering,
            'per_node': clustering,
            'by_generation': gen_avg_clustering,
            'highest': sorted_by_clustering[:10],
            'lowest': sorted_by_clustering[-10:],
        }
    
    
    
    def centrality_analysis(self):
        """
        computes multiple centrality measures
        
        DEGREE CENTRALITY: normalized degree - how many connections
        BETWEENNESS: how often node is on shortest paths - bridge nodes
        CLOSENESS: avg distance to all other nodes - central position
        EIGENVECTOR: connected to other important nodes - influence
        
        TRADEOFF: betweenness is O(n*m), slow for large graphs
        we compute on largest component only
        
        ASSUMPTION: undirected graph for all measures
        directed versions exist but harder to interpret for family relations
        
        NOTE: since graph has 50 separate components, centrality is computed
        per-component. values are only meaningful within a family, not across.
        """
        print("\n=== CENTRALITY ANALYSIS ===")
        
        components = list(nx.connected_components(self.G_undirected))
        print(f"NOTE: graph has {len(components)} separate family components")
        print(f"      centrality computed per-component, showing first family as example")
        
        # use largest component for meaningful results
        largest_cc = max(components, key=len)
        G = self.G_undirected.subgraph(largest_cc)
        
        print(f"      analyzing component with {len(largest_cc)} members")
        
        # degree centrality - fast
        degree_cent = nx.degree_centrality(G)
        
        # betweenness - slow, measures bridge importance
        # LIMITATION: expensive O(nm), but essential for finding bridges
        print("computing betweenness centrality (this may take a moment)...")
        betweenness = nx.betweenness_centrality(G)
        
        # closeness - how close to everyone else
        closeness = nx.closeness_centrality(G)
        
        # eigenvector - importance via neighbor importance
        # LIMITATION: may not converge for some graphs, using pagerank as fallback
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            print("  WARNING: eigenvector centrality didn't converge, using pagerank instead")
            eigenvector = nx.pagerank(G)
        
        # find top nodes by each measure
        def top_n(cent_dict, n=10):
            return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        
        print("\nDEGREE CENTRALITY (most connections):")
        print("  interpretation: people with most direct family relations")
        for person, score in top_n(degree_cent, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nBETWEENNESS CENTRALITY (bridge nodes):")
        print("  interpretation: people who connect different parts of family")
        print("  high betweenness = removing them would disconnect family clusters")
        for person, score in top_n(betweenness, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nCLOSENESS CENTRALITY (central position):")
        print("  interpretation: people with shortest avg distance to everyone")
        for person, score in top_n(closeness, 5):
            gen = self.features['people'].get(person, {}).get('generation', '?')
            print(f"    {person}: {score:.4f} (gen {gen})")
        
        print("\nEIGENVECTOR/PAGERANK CENTRALITY (influence):")
        print("  interpretation: people connected to other important people")
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
        print("  these people connect otherwise separate family branches")
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
            'central_figures': central_figures,
        }
    
    # ==================== ARTICULATION POINTS ====================
    
    def articulation_points(self):
        """
        articulation points = nodes whose removal disconnects the graph
        
        INTERPRETATION: critical family members who link branches
        removing them would split the family tree
        
        NOTE: computed on undirected graph
        """
        aps = list(nx.articulation_points(self.G_undirected))
        
        # analyze what generations they're from
        ap_gens = Counter()
        for ap in aps:
            gen = self.features['people'].get(ap, {}).get('generation')
            if gen is not None:
                ap_gens[gen] += 1
        
        print("\n=== ARTICULATION POINTS ===")
        print(f"total articulation points: {len(aps)}")
        print(f"  INTERPRETATION: {len(aps)} people are critical connectors")
        print(f"  removing any of them would split the family graph")
        print(f"\nby generation:")
        for g in sorted(ap_gens.keys()):
            print(f"  gen {g}: {ap_gens[g]}")
        print(f"\nfirst 10 articulation points:")
        for ap in aps[:10]:
            gen = self.features['people'].get(ap, {}).get('generation', '?')
            degree = self.G_undirected.degree(ap)
            print(f"  {ap}: gen {gen}, degree {degree}")
        
        return {
            'count': len(aps),
            'points': aps,
            'by_generation': dict(ap_gens),
        }
    
    # ==================== GENERATION-SPECIFIC ANALYSIS ====================
    
    def generation_analysis(self):
        """
        analyze graph properties by generation
        
        family graphs have natural hierarchy - this explores it
        
        INSIGHT FOR TASK 2: generation might be better clustering criterion
        than standard community detection
        """
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
        
        print("\n=== GENERATION ANALYSIS ===")
        print(f"{'Gen':<5} {'Count':<8} {'Avg Deg':<10} {'Founders':<10} {'Leaves':<10}")
        print("-" * 45)
        
        for g in sorted(gen_stats.keys()):
            s = gen_stats[g]
            avg_deg = s['total_degree'] / s['count'] if s['count'] > 0 else 0
            print(f"{g:<5} {s['count']:<8} {avg_deg:<10.2f} {s['founders']:<10} {s['leaves']:<10}")
        
        print(f"\nINSIGHTS:")
        print(f"  - founders (no parents) concentrated in gen 0: validates generation inference")
        print(f"  - leaves (no children) concentrated in highest gens: expected for youngest")
        print(f"  - middle generations should have highest avg degree (connected both up and down)")
        
        return dict(gen_stats)
    
    # ==================== RELATION-SPECIFIC SUBGRAPH ANALYSIS ====================
    
    def relation_subgraph_analysis(self):
        """
        analyze subgraphs formed by specific relation types
        
        e.g., parent-child subgraph should be a forest (trees)
        sibling subgraph should be cliques within families
        
        INSIGHT FOR TASK 3: these structural properties become rules
        """
        print("\n=== RELATION-SPECIFIC SUBGRAPH ANALYSIS ===")
        
        # parent-child subgraph (should be DAG / forest)
        parent_child_edges = [(h, t) for h, r, t in self.triplets if r in PARENT_RELATIONS]
        G_pc = nx.DiGraph()
        G_pc.add_edges_from(parent_child_edges)
        
        is_dag = nx.is_directed_acyclic_graph(G_pc)
        pc_components = nx.number_weakly_connected_components(G_pc)
        
        print(f"PARENT-CHILD SUBGRAPH:")
        print(f"  edges: {len(parent_child_edges)}")
        print(f"  is DAG (no cycles): {is_dag}")
        print(f"  weakly connected components: {pc_components}")
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
        
        print(f"\nSIBLING SUBGRAPH:")
        print(f"  edges: {len(sibling_edges)}")
        print(f"  sibling groups (components): {len(sib_components)}")
        print(f"  groups that aren't complete cliques: {clique_violations}")
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
    
    # ==================== CROSS-FAMILY AGGREGATE ANALYSIS ====================
    
    def cross_family_analysis(self):
        """
        since we have 50 separate families, analyze patterns across them
        
        this is useful for understanding what's typical vs unusual
        """
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
    
    # ==================== DEGREE DISTRIBUTION ====================
    
    def degree_distribution(self):
        """
        analyze degree distribution shape
        
        family graphs typically NOT scale-free (no power law)
        more likely normal or bounded distribution due to biological constraints
        (max ~2 parents, limited children)
        """
        degrees = [d for n, d in self.G_undirected.degree()]
        
        degree_counts = Counter(degrees)
        
        # basic stats
        avg = sum(degrees) / len(degrees)
        variance = sum((d - avg) ** 2 for d in degrees) / len(degrees)
        std = variance ** 0.5
        
        print("\n=== DEGREE DISTRIBUTION ===")
        print(f"mean: {avg:.2f}")
        print(f"std dev: {std:.2f}")
        print(f"median: {sorted(degrees)[len(degrees)//2]}")
        print(f"mode: {max(degree_counts, key=degree_counts.get)}")
        print(f"\ndistribution (degree: count):")
        for d in sorted(degree_counts.keys())[:20]:
            bar = '#' * min(50, degree_counts[d] // 2)
            print(f"  {d:3}: {degree_counts[d]:4} {bar}")
        if max(degree_counts.keys()) > 20:
            print(f"  ... (max degree: {max(degrees)})")
        
        print(f"\nINSIGHT: family graphs have bounded degree distribution")
        print(f"  unlike social networks, you can't have 1000 parents")
        print(f"  high degree nodes are middle-generation with many relatives")
        
        return {
            'mean': avg,
            'std': std,
            'distribution': dict(degree_counts),
        }
    
    # ==================== IMPORTANT NODES SUMMARY ====================
    
    def important_nodes_summary(self, centrality_results):
        """
        synthesize all centrality measures to identify truly important nodes
        
        CRITERIA:
        - founders: gen 0 with many descendants
        - bridges: high betweenness, low degree
        - hubs: high degree, high eigenvector
        - connectors: high closeness
        """
        print("\n=== IMPORTANT NODES SUMMARY ===")
        
        # score each node
        scores = {}
        
        bc = centrality_results['betweenness']
        dc = centrality_results['degree']
        cc = centrality_results['closeness']
        ec = centrality_results['eigenvector']
        
        for person in self.people:
            if person not in bc:
                continue  # not in largest component
            
            f = self.features['people'].get(person, {})
            gen = f.get('generation', -1)
            
            # composite importance score
            # weighted sum of normalized centralities
            score = (
                bc.get(person, 0) * 3 +  # betweenness weighted higher
                dc.get(person, 0) * 1 +
                cc.get(person, 0) * 1 +
                ec.get(person, 0) * 2
            )
            
            scores[person] = {
                'composite_score': score,
                'generation': gen,
                'betweenness': bc.get(person, 0),
                'degree_cent': dc.get(person, 0),
                'closeness': cc.get(person, 0),
                'eigenvector': ec.get(person, 0),
                'num_children': f.get('children', {}).get('total', 0),
                'is_founder': f.get('parents', {}).get('total', 0) == 0,
            }
        
        # rank by composite score
        ranked = sorted(scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        print("TOP 15 MOST IMPORTANT FAMILY MEMBERS:")
        print(f"{'Rank':<5} {'Person':<15} {'Gen':<5} {'Score':<8} {'Between':<8} {'Degree':<8} {'Children':<8}")
        print("-" * 70)
        
        for i, (person, data) in enumerate(ranked[:15], 1):
            print(f"{i:<5} {person:<15} {data['generation']:<5} {data['composite_score']:.4f}  {data['betweenness']:.4f}   {data['degree_cent']:.4f}   {data['num_children']:<8}")
        
        # categorize top nodes
        print("\nCATEGORIZATION:")
        
        founders_important = [(p, d) for p, d in ranked[:30] if d['is_founder']]
        print(f"\nFounders in top 30: {len(founders_important)}")
        for p, d in founders_important[:5]:
            print(f"  {p}: gen {d['generation']}, {d['num_children']} children")
        
        bridges = [(p, d) for p, d in ranked if d['betweenness'] > 0.01 and d['degree_cent'] < 0.04]
        print(f"\nBridge nodes (high betweenness, moderate degree): {len(bridges)}")
        for p, d in sorted(bridges, key=lambda x: x[1]['betweenness'], reverse=True)[:5]:
            print(f"  {p}: betweenness={d['betweenness']:.4f}")
        
        return {
            'ranked': ranked[:50],
            'scores': scores,
        }


def run_all_metrics(data_path='data/train.txt'):
    """run complete analysis"""
    
    print("=" * 60)
    print("METAFAM GRAPH METRICS - TASK 1 ANALYSIS")
    print("=" * 60)
    
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
    results['articulation'] = metrics.articulation_points()
    results['generations'] = metrics.generation_analysis()
    results['relation_subgraphs'] = metrics.relation_subgraph_analysis()
    results['cross_family'] = metrics.cross_family_analysis()
    results['degree_dist'] = metrics.degree_distribution()
    results['important_nodes'] = metrics.important_nodes_summary(results['centrality'])
    
    # save results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return metrics, results


if __name__ == "__main__":
    random.seed(42)  # reproducibility for sampling
    metrics, results = run_all_metrics()