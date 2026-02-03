import networkx as nx
from collections import defaultdict, Counter
import pickle
import os
import sys

# add parent dir to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MetaFAMLoader
from src.feature_extractor import RawFeatureExtractor
from src.inference import infer_gender, classify_anomaly_severity, is_leaf_node, is_founder_node
from src.constants import GENERATION_DELTAS, PARENT_RELATIONS, SIBLING_RELATIONS


class DashboardData:
    """
    handles all data loading and provides clean interfaces for dashboard
    """
    
    def __init__(self):
        self.triplets = []
        self.people = set()
        self.relation_types = set()
        self.features = {}  # raw features from extractor
        self.node_data = {}  # processed for dashboard use
        self.G = None  # networkx graph
        
        # indexes for fast lookup
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.edge_set = set()
    
    def load(self, filepath: str):
        """load and process data"""
        
        # use our loader
        loader = MetaFAMLoader(filepath)
        loader.load()
        
        self.triplets = loader.triplets
        self.people = loader.people
        self.relation_types = loader.relation_types
        
        # build indexes
        for h, r, t in self.triplets:
            self.outgoing[h].append((r, t))
            self.incoming[t].append((r, h))
            self.edge_set.add((h, r, t))
        
        # extract features using our correct extractor
        extractor = RawFeatureExtractor(self.triplets, self.people)
        self.features = extractor.extract_all()
        
        # build networkx graph
        self._build_graph()
        
        # process into dashboard-friendly format
        self._process_for_dashboard()
        
        return self
    
    def _build_graph(self):
        """build networkx multigraph (allows multiple edges between same nodes)"""
        self.G = nx.MultiDiGraph()
        
        for h, r, t in self.triplets:
            self.G.add_edge(h, t, relation=r)
        
        # also build simple graph for centrality computations
        self.G_simple = nx.Graph()
        for h, r, t in self.triplets:
            self.G_simple.add_edge(h, t)
        
        # compute centralities (cached)
        self._centralities = None
    
    def get_centralities(self, force_recompute=False):
        """compute and cache centrality measures"""
        if self._centralities is not None and not force_recompute:
            return self._centralities
        
        self._centralities = {}
        
        # degree centrality
        deg_cent = nx.degree_centrality(self.G_simple)
        
        # betweenness - can be slow for large graphs
        if len(self.people) < 5000:
            between_cent = nx.betweenness_centrality(self.G_simple)
        else:
            # approximate for large graphs
            between_cent = nx.betweenness_centrality(self.G_simple, k=min(500, len(self.people)))
        
        # closeness
        close_cent = nx.closeness_centrality(self.G_simple)
        
        # pagerank on directed graph
        pagerank = nx.pagerank(self.G)
        
        for person_id in self.people:
            self._centralities[person_id] = {
                'degree_centrality': deg_cent.get(person_id, 0),
                'betweenness': between_cent.get(person_id, 0),
                'closeness': close_cent.get(person_id, 0),
                'pagerank': pagerank.get(person_id, 0),
            }
        
        return self._centralities
    
    def get_node_centrality(self, person_id):
        """get centrality for single node"""
        cents = self.get_centralities()
        return cents.get(person_id, {})
    
    def find_paths(self, person_a: str, person_b: str, max_hops: int = 3) -> list:
        """
        find all simple paths between two people up to max_hops
        returns list of paths, each path is list of (person, relation, person) tuples
        """
        if person_a not in self.G or person_b not in self.G:
            return []
        
        paths = []
        
        # use simple graph for path finding (undirected)
        try:
            simple_paths = list(nx.all_simple_paths(
                self.G_simple, person_a, person_b, cutoff=max_hops
            ))
        except nx.NetworkXNoPath:
            return []
        
        # for each path, find the actual relations
        for path in simple_paths:
            path_with_relations = []
            for i in range(len(path) - 1):
                h, t = path[i], path[i+1]
                # find relation between h and t
                rel = self._get_relation_between(h, t)
                path_with_relations.append({
                    'from': h,
                    'relation': rel,
                    'to': t
                })
            paths.append(path_with_relations)
        
        return paths
    
    def _get_relation_between(self, person_a: str, person_b: str) -> str:
        """get relation between two people (either direction)"""
        # check outgoing from a
        for r, t in self.outgoing[person_a]:
            if t == person_b:
                return r
        # check outgoing from b (reverse)
        for r, t in self.outgoing[person_b]:
            if t == person_a:
                return f"←{r}"
        return "connected"
    
    def get_path_subgraph(self, paths: list) -> dict:
        """convert paths to nodes and edges for visualization"""
        nodes = set()
        edges = []
        seen_edges = set()
        
        for path in paths:
            for step in path:
                nodes.add(step['from'])
                nodes.add(step['to'])
                edge_key = (step['from'], step['relation'], step['to'])
                if edge_key not in seen_edges:
                    edges.append(step)
                    seen_edges.add(edge_key)
        
        return {'nodes': list(nodes), 'edges': edges}
    
    def get_interesting_people(self, metric: str, n: int = 10) -> list:
        """
        find interesting people by various metrics
        metric: 'degree', 'betweenness', 'founders', 'gen_zscore', 'bridges'
        """
        if metric == 'degree':
            return self.get_high_degree_nodes(n)
        
        elif metric == 'betweenness':
            cents = self.get_centralities()
            sorted_by_between = sorted(
                cents.items(),
                key=lambda x: x[1]['betweenness'],
                reverse=True
            )[:n]
            return [
                {**self.node_data[p], 'betweenness': c['betweenness']}
                for p, c in sorted_by_between
            ]
        
        elif metric == 'founders':
            founders = self.get_founders()
            # sort by number of descendants (children + grandchildren)
            founder_data = []
            for f in founders:
                node = self.node_data[f]
                descendants = node['num_children']
                for child in node.get('children', []):
                    if child in self.node_data:
                        descendants += self.node_data[child]['num_children']
                founder_data.append({**node, 'descendants': descendants})
            return sorted(founder_data, key=lambda x: x['descendants'], reverse=True)[:n]
        
        elif metric == 'gen_zscore':
            # most central for their generation
            from collections import defaultdict
            import statistics
            
            gen_degrees = defaultdict(list)
            for p, node in self.node_data.items():
                gen = node['generation']
                if gen >= 0:
                    gen_degrees[gen].append((p, node['degree']))
            
            results = []
            for gen, degrees in gen_degrees.items():
                if len(degrees) < 2:
                    continue
                degs = [d for _, d in degrees]
                mean_deg = statistics.mean(degs)
                std_deg = statistics.stdev(degs) if len(degs) > 1 else 1
                
                for p, deg in degrees:
                    zscore = (deg - mean_deg) / std_deg if std_deg > 0 else 0
                    if zscore > 1.5:  # significantly above average
                        results.append({
                            **self.node_data[p],
                            'zscore': round(zscore, 2),
                            'gen_avg_degree': round(mean_deg, 1)
                        })
            
            return sorted(results, key=lambda x: x['zscore'], reverse=True)[:n]
        
        elif metric == 'bridges':
            # nodes that connect different components if removed
            cents = self.get_centralities()
            # high betweenness + low clustering = bridge
            results = []
            for p, c in cents.items():
                if c['betweenness'] > 0.01:  # threshold
                    node = self.node_data[p]
                    results.append({
                        **node,
                        'betweenness': c['betweenness'],
                    })
            return sorted(results, key=lambda x: x['betweenness'], reverse=True)[:n]
        
        return []
    
    def get_families(self) -> list:
        """get all connected components (families)"""
        G_undirected = self.G.to_undirected()
        components = list(nx.connected_components(G_undirected))
        
        families = []
        for i, comp in enumerate(sorted(components, key=len, reverse=True)):
            families.append({
                'family_id': i,
                'size': len(comp),
                'members': list(comp)[:10],  # just first 10 for display
            })
        
        return families
    
    def get_subgraph_stats(self, nodes: list) -> dict:
        """compute stats for a subgraph"""
        node_set = set(nodes)
        
        # filter node data
        sub_nodes = [self.node_data[p] for p in nodes if p in self.node_data]
        
        if not sub_nodes:
            return {}
        
        # edges within subgraph
        sub_edges = [
            (h, r, t) for h, r, t in self.triplets
            if h in node_set and t in node_set
        ]
        
        degrees = [n['degree'] for n in sub_nodes]
        gens = [n['generation'] for n in sub_nodes if n['generation'] >= 0]
        
        return {
            'num_nodes': len(sub_nodes),
            'num_edges': len(sub_edges),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'num_founders': sum(1 for n in sub_nodes if n['is_founder']),
            'num_leaves': sum(1 for n in sub_nodes if n['is_leaf']),
            'num_anomalies': sum(1 for n in sub_nodes if n['has_anomaly']),
            'generation_range': (min(gens), max(gens)) if gens else (0, 0),
            'gender_counts': Counter(n['gender'] for n in sub_nodes),
        }
    
    def get_full_graph_stats(self) -> dict:
        """compute stats for full graph"""
        families = self.get_families()
        degrees = [n['degree'] for n in self.node_data.values()]
        
        return {
            'num_nodes': len(self.people),
            'num_edges': len(self.triplets),
            'num_families': len(families),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'num_relation_types': len(self.relation_types),
        }
    
    def _process_for_dashboard(self):
        """convert raw features to dashboard-friendly flat format"""
        
        for person_id, f in self.features['people'].items():
            gender_inf = infer_gender(f['gender_evidence'])
            anomaly_class = classify_anomaly_severity(f['anomalies'])
            
            # combine all parents, resolve gender where possible
            mothers = list(f['parents']['mothers'])
            fathers = list(f['parents']['fathers'])
            
            # for unknown_gender_parents, infer their gender
            for parent in f['parents'].get('unknown_gender_parents', []):
                if parent in self.features['people']:
                    parent_ev = self.features['people'][parent]['gender_evidence']
                    parent_gender = infer_gender(parent_ev)
                    if parent_gender['gender'] == 'F':
                        mothers.append(parent)
                    elif parent_gender['gender'] == 'M':
                        fathers.append(parent)
                    else:
                        # truly unknown, just add to mothers arbitrarily
                        mothers.append(parent)
            
            self.node_data[person_id] = {
                'person_id': person_id,
                'gender': gender_inf['gender'],
                'gender_confidence': gender_inf['confidence'],
                'generation': f['generation'] if f['generation'] is not None else -1,
                'degree': f['relation_counts']['total'],
                'in_degree': f['relation_counts']['total_in'],
                'out_degree': f['relation_counts']['total_out'],
                'num_parents': len(mothers) + len(fathers),
                'num_children': f['children']['total'],
                'num_siblings': len(f['siblings']),
                'mothers': mothers,
                'fathers': fathers,
                'children': f['children']['children'],
                'siblings': f['siblings'],
                'is_founder': is_founder_node(f),
                'is_leaf': is_leaf_node(f),
                'has_anomaly': anomaly_class['has_anomalies'],
                'anomalies': anomaly_class.get('types', []),
                'anomaly_severity': anomaly_class.get('max_severity', 0),
            }
    
    def get_node(self, person_id: str) -> dict:
        """get processed node data"""
        return self.node_data.get(person_id)
    
    def get_all_nodes(self) -> dict:
        """get all node data"""
        return self.node_data
    
    def get_ego_network(self, person_id: str, hops: int = 1) -> dict:
        """
        get subgraph around a person
        returns nodes and edges for visualization
        """
        if person_id not in self.G:
            return {'nodes': [], 'edges': [], 'center': person_id}
        
        nodes = {person_id}
        frontier = {person_id}
        
        for _ in range(hops):
            new_frontier = set()
            for p in frontier:
                # successors and predecessors
                new_frontier.update(self.G.successors(p))
                new_frontier.update(self.G.predecessors(p))
            nodes.update(new_frontier)
            frontier = new_frontier
        
        # collect edges within this subgraph
        edges = []
        seen_edges = set()
        for h, r, t in self.triplets:
            if h in nodes and t in nodes:
                edge_key = (h, r, t)
                if edge_key not in seen_edges:
                    edges.append({'from': h, 'relation': r, 'to': t})
                    seen_edges.add(edge_key)
        
        return {
            'center': person_id,
            'nodes': list(nodes),
            'edges': edges,
        }
    
    def get_family_tree(self, person_id: str) -> dict:
        """
        get vertical family tree (parents, grandparents, children, grandchildren)
        more structured than ego network
        """
        if person_id not in self.node_data:
            return {'nodes': [], 'edges': [], 'center': person_id}
        
        nodes = {person_id}
        edges = []
        
        node = self.node_data[person_id]
        
        # parents (using processed node_data which has resolved genders)
        for m in node.get('mothers', []):
            nodes.add(m)
            edges.append({'from': m, 'relation': 'motherOf', 'to': person_id})
        for fa in node.get('fathers', []):
            nodes.add(fa)
            edges.append({'from': fa, 'relation': 'fatherOf', 'to': person_id})
        
        # grandparents (parents of parents)
        all_parents = node.get('mothers', []) + node.get('fathers', [])
        for parent in all_parents:
            if parent in self.node_data:
                parent_node = self.node_data[parent]
                for gm in parent_node.get('mothers', []):
                    nodes.add(gm)
                    edges.append({'from': gm, 'relation': 'motherOf', 'to': parent})
                for gf in parent_node.get('fathers', []):
                    nodes.add(gf)
                    edges.append({'from': gf, 'relation': 'fatherOf', 'to': parent})
        
        # siblings
        for sib in node.get('siblings', []):
            nodes.add(sib)
            # find actual relation in triplets
            found = False
            for h, r, t in self.triplets:
                if h == person_id and t == sib and r in SIBLING_RELATIONS:
                    edges.append({'from': person_id, 'relation': r, 'to': sib})
                    found = True
                    break
                if h == sib and t == person_id and r in SIBLING_RELATIONS:
                    edges.append({'from': sib, 'relation': r, 'to': person_id})
                    found = True
                    break
            if not found:
                edges.append({'from': person_id, 'relation': 'siblingOf', 'to': sib})
        
        # children
        for child in node.get('children', []):
            nodes.add(child)
            # determine relation based on person's gender
            if node['gender'] == 'F':
                edges.append({'from': person_id, 'relation': 'motherOf', 'to': child})
            else:
                edges.append({'from': person_id, 'relation': 'fatherOf', 'to': child})
            
            # grandchildren
            if child in self.node_data:
                child_node = self.node_data[child]
                for gc in child_node.get('children', []):
                    nodes.add(gc)
                    if child_node['gender'] == 'F':
                        edges.append({'from': child, 'relation': 'motherOf', 'to': gc})
                    else:
                        edges.append({'from': child, 'relation': 'fatherOf', 'to': gc})
        
        return {
            'center': person_id,
            'nodes': list(nodes),
            'edges': edges,
        }
    
    def get_connected_component(self, person_id: str) -> set:
        """get all people in same connected component"""
        if person_id not in self.G:
            return set()
        
        G_undirected = self.G.to_undirected()
        for comp in nx.connected_components(G_undirected):
            if person_id in comp:
                return comp
        
        return {person_id}
    
    def get_generation_stats(self) -> dict:
        """generation distribution"""
        gens = [n['generation'] for n in self.node_data.values()]
        counts = Counter(gens)
        return dict(sorted(counts.items()))
    
    def get_gender_stats(self) -> dict:
        """gender distribution"""
        genders = [n['gender'] for n in self.node_data.values()]
        return dict(Counter(genders))
    
    def get_relation_stats(self) -> dict:
        """relation type counts"""
        rels = [r for _, r, _ in self.triplets]
        return dict(Counter(rels).most_common())
    
    def get_degree_stats(self) -> dict:
        """degree distribution summary"""
        degrees = [n['degree'] for n in self.node_data.values()]
        return {
            'min': min(degrees),
            'max': max(degrees),
            'avg': sum(degrees) / len(degrees),
            'distribution': dict(Counter(degrees)),
        }
    
    def search_people(self, query: str, limit: int = 20) -> list:
        """simple search by person id prefix"""
        query = query.lower()
        matches = [p for p in self.people if query in p.lower()]
        return sorted(matches)[:limit]
    
    def get_high_degree_nodes(self, n: int = 20) -> list:
        """get top n nodes by degree"""
        sorted_nodes = sorted(self.node_data.items(), key=lambda x: x[1]['degree'], reverse=True)
        return [{'person_id': p, **d} for p, d in sorted_nodes[:n]]
    
    def get_founders(self) -> list:
        """get all founder nodes"""
        return [p for p, d in self.node_data.items() if d['is_founder']]
    
    def get_anomalous_nodes(self) -> list:
        """get nodes with anomalies"""
        return [{'person_id': p, **d} for p, d in self.node_data.items() if d['has_anomaly']]
    
    def categorize_relation(self, relation: str) -> str:
        """categorize a relation type"""
        if relation in GENERATION_DELTAS:
            delta = GENERATION_DELTAS[relation]
            if delta == 0:
                return 'horizontal'
            else:
                return 'vertical'
        return 'unknown'
    
    def save_cache(self, filepath: str = 'dashboard_cache.pkl'):
        
        data = {
            'triplets': self.triplets,
            'people': self.people,
            'relation_types': self.relation_types,
            'features': self.features,
            'node_data': self.node_data,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_cache(self, filepath: str = 'dashboard_cache.pkl') -> bool:
        """load from cache if exists"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.triplets = data['triplets']
        self.people = data['people']
        self.relation_types = data['relation_types']
        self.features = data['features']
        self.node_data = data['node_data']
        
        # rebuild indexes
        for h, r, t in self.triplets:
            self.outgoing[h].append((r, t))
            self.incoming[t].append((r, h))
            self.edge_set.add((h, r, t))
        
        self._build_graph()
        
        return True