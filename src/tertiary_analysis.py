# extra insights and stuff that feeds into dashboard and qualitative insights

import json
import random
from collections import defaultdict, Counter
from itertools import combinations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MetaFAMLoader
from src.feature_extractor import RawFeatureExtractor
from src.constants import  *


class Explorer:
    
    def __init__(self, triplets, features):
        self.triplets = triplets
        self.features = features
        self.edge_set = set((h, r, t) for h, r, t in triplets)
        
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for h, r, t in triplets:
            self.outgoing[h].append((r, t))
            self.incoming[t].append((r, h))
        
        # gonna need this a lot for visualizations
        self.people = list(features['people'].keys())

    
    def derivability_extended(self):
       
        # check which relations can be derived from simpler ones
        results = {}
        
        # grandparent: already done but redo here for completeness

        for gp_rel, parent_gender in [('grandmotherOf', 'motherOf'), ('grandfatherOf', 'fatherOf')]:
            total = 0
            derivable = 0

            for h, r, t in self.triplets:
            
                if r != gp_rel:
                    continue
                total += 1
            
                # check if h -[motherOf/fatherOf]-> mid -[parentOf]-> t
                for r1, mid in self.outgoing[h]:
                    if r1 == parent_gender:
                        for r2, target in self.outgoing[mid]:
                            if r2 in PARENT_RELATIONS and target == t:
                                derivable += 1
                                break
                        else:
                            continue
                        break
            
            results[gp_rel] = {
                'total': total,
                'derivable': derivable,
                'pct': round(derivable / total * 100, 1) if total > 0 else 0
            }
        
        # aunt/uncle: auntOf(A,C) = sisterOf(A,B) + parentOf(B,C)
        # i.e A is sister of B, and B is parent of C
        for au_rel, sib_rel in [('auntOf', 'sisterOf'), ('uncleOf', 'brotherOf')]:
            total = 0
            derivable = 0
            for h, r, t in self.triplets:
                if r != au_rel:
                    continue
                total += 1
                found = False
                # h is aunt/uncle of t
                # so h should be sibling of t's parent
                for r1, sibling in self.outgoing[h]:
                    if r1 == sib_rel:
                        # sibling should be parent of t
                        for r2, child in self.outgoing[sibling]:
                            if r2 in PARENT_RELATIONS and child == t:
                                found = True
                                break
                    if found:
                        break
                if found:
                    derivable += 1
            
            results[au_rel] = {
                'total': total,
                'derivable': derivable,
                'pct': round(derivable / total * 100, 1) if total > 0 else 0
            }
        
        # niece/nephew: nieceOf(A,C) = daughterOf(A,B) + siblingOf(B,C)
        
        # so: A -[childOf]-> B -[siblingOf]-> C
        for nn_rel, child_rel in [('nieceOf', 'daughterOf'), ('nephewOf', 'sonOf')]:
            total = 0
            derivable = 0
            for h, r, t in self.triplets:
                if r != nn_rel:
                    continue
                total += 1
                found = False
                # h is niece of t, so h's parent is sibling of t

                for r1, parent in self.outgoing[h]:
                
                    if r1 == child_rel:
                        # parent should be sibling of t
                        for r2, sib in self.outgoing[parent]:
                            if r2 in SIBLING_RELATIONS and sib == t:
                
                                found = True
                                break
                        # also check incoming sibling relations
                
                        if not found:
                            for r2, sib in self.incoming[parent]:
                                if r2 in SIBLING_RELATIONS and sib == t:
                                    found = True
                                    break
                    if found:
                        break
                
                if found:
                    derivable += 1
            
            results[nn_rel] = {
                
                'total': total,
                'derivable': derivable,
                'pct': round(derivable / total * 100, 1) if total > 0 else 0
            }
        
        # cousin: cousinOf(A,B) = A and B share a grandparent but not a parent
        # this ones messier, i have skipped exact derivation
        
        for cuz_rel in ['girlCousinOf', 'boyCousinOf']:
            total = sum(1 for h, r, t in self.triplets if r == cuz_rel)
            results[cuz_rel] = {
                'total': total,
                'derivable': 'not computed - complex chain',
                'pct': None,
                'note': 'would need grandparent + sibling + parent chain'
            }
        
        return results
    
    
    def generation_normalized_degree(self):
        
        # IMPORTANT FIND HERE !!!
        # z-score degree within each generation
        # raw degree is useless bc gen 1 people naturally have more connections
        # group by generation


        gen_to_people = defaultdict(list)
        for person_id, f in self.features['people'].items():
            gen = f['generation']
            if gen is not None:
                degree = f['relation_counts']['total']
                gen_to_people[gen].append((person_id, degree))
        
        # compute mean/std per generation
        gen_stats = {}
        for gen, people in gen_to_people.items():
            degrees = [d for _, d in people]
            mean = sum(degrees) / len(degrees)
            variance = sum((d - mean) ** 2 for d in degrees) / len(degrees)
            std = variance ** 0.5 if variance > 0 else 1  # avoid div by 0
            gen_stats[gen] = {'mean': mean, 'std': std, 'n': len(degrees)}
        
        # compute z-scores
        z_scores = {}
        for gen, people in gen_to_people.items():
            mean = gen_stats[gen]['mean']
            std = gen_stats[gen]['std']
            for person_id, degree in people:
                z = (degree - mean) / std if std > 0 else 0
                z_scores[person_id] = {
                    'generation': gen,
                    'raw_degree': degree,
                    'z_score': round(z, 2),
                    'gen_mean': round(mean, 1),
                    'gen_std': round(std, 1),
                }
        
        # find outliers (|z| > 2)
        outliers_high = [(p, d) for p, d in z_scores.items() if d['z_score'] > 2]
        outliers_low = [(p, d) for p, d in z_scores.items() if d['z_score'] < -2]
        
        return {
            'gen_stats': gen_stats,
            'z_scores': z_scores,
            'outliers_high': sorted(outliers_high, key=lambda x: x[1]['z_score'], reverse=True)[:20],
            'outliers_low': sorted(outliers_low, key=lambda x: x[1]['z_score'])[:20],
        }
    
    
    def path_multiplicity_sample(self, n_pairs=100, max_hops=3):
    
        # for random pairs, count how many distinct paths connect them
        # high multiplicity = redundant connection (robust)
        # low multiplicity = fragile connection
        
        # this fried my cpu cause its expensive so we sample that are actually connected (within same component)
        # for now just random pairs and see what we get
        
        random.seed(42)  # reproduciblity
        sampled_pairs = []
        
        attempts = 0
        while len(sampled_pairs) < n_pairs and attempts < n_pairs * 10:
            a, b = random.sample(self.people, 2)
            sampled_pairs.append((a, b))
            attempts += 1
        
        results = []
        for a, b in sampled_pairs:
            paths = self._find_paths(a, b, max_hops)
            results.append({
                'from': a,
                'to': b,
                'num_paths': len(paths),
                'paths': paths[:5],  # keep max 5 for storage
                'shortest': min(len(p) for p in paths) if paths else None,
            })
        
        # summarize
        path_counts = [r['num_paths'] for r in results]
        connected = [r for r in results if r['num_paths'] > 0]
        
        return {
            'sampled': len(results),
            'connected_pairs': len(connected),
            'avg_paths_when_connected': round(sum(r['num_paths'] for r in connected) / len(connected), 2) if connected else 0,
            'max_paths': max(path_counts) if path_counts else 0,
            'distribution': dict(Counter(path_counts)),
            'examples_high_multiplicity': sorted([r for r in results if r['num_paths'] >= 3], key=lambda x: x['num_paths'], reverse=True)[:10],
            'examples_single_path': [r for r in results if r['num_paths'] == 1][:10],
        }
    
    def _find_paths(self, start, end, max_hops):
        
        # using bfs to find all paths up to max_hops
        
        if start == end:
            return []
        
        # paths are list of (relation, node) tuples
        queue = [(start, [])]
        found_paths = []
        visited_states = set()  # (current_node, frozenset of visited) to avoid cycles
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
            
            # check outgoing
            for rel, neighbor in self.outgoing[current]:
                if neighbor == end:
                    found_paths.append(path + [(rel, neighbor)])
                elif neighbor not in [p[1] for p in path] and neighbor != start:
                    # dont revisit nodes in current path
                    state = (neighbor, frozenset(p[1] for p in path))
                    if state not in visited_states:
                        visited_states.add(state)
                        queue.append((neighbor, path + [(rel, neighbor)]))
            
            # check incoming (reverse direction)
            for rel, neighbor in self.incoming[current]:
                if neighbor == end:
                    found_paths.append(path + [(f"inv_{rel}", neighbor)])
                elif neighbor not in [p[1] for p in path] and neighbor != start:
                    state = (neighbor, frozenset(p[1] for p in path))
                    if state not in visited_states:
                        visited_states.add(state)
                        queue.append((neighbor, path + [(f"inv_{rel}", neighbor)]))
            
            # cap to avoid explosion
            if len(found_paths) > 20:
                break
        
        return found_paths
    
    # ============ RELATION CO-OCCURRENCE ============
    
    def relation_cooccurrence(self):
        """
        which outgoing relations tend to appear together on same person
        e.g., motherOf usually comes with daughterOf (youre someones mom and someones daughter)
        """
        # for each person, get set of outgoing relation types
        person_relations = {}
        for person_id in self.people:
            rels = set(r for r, _ in self.outgoing[person_id])
            person_relations[person_id] = rels
        
        # count co-occurrences
        rel_types = list(set(r for rels in person_relations.values() for r in rels))
        cooccur = defaultdict(int)
        rel_counts = Counter()
        
        for person_id, rels in person_relations.items():
            for r in rels:
                rel_counts[r] += 1
            for r1, r2 in combinations(sorted(rels), 2):
                cooccur[(r1, r2)] += 1
        
        # compute lift (observed / expected)
        # lift > 1 means they appear together more than chance
        n_people = len(self.people)
        lift_scores = {}
        for (r1, r2), count in cooccur.items():
            expected = (rel_counts[r1] / n_people) * (rel_counts[r2] / n_people) * n_people
            lift = count / expected if expected > 0 else 0
            lift_scores[(r1, r2)] = {
                'count': count,
                'expected': round(expected, 1),
                'lift': round(lift, 2),
            }
        
        # sort by lift
        sorted_by_lift = sorted(lift_scores.items(), key=lambda x: x[1]['lift'], reverse=True)
        
        # also find pairs that never co-occur but could
        never_cooccur = []
        for r1 in rel_types:
            for r2 in rel_types:
                if r1 < r2 and (r1, r2) not in cooccur:
                    if rel_counts[r1] > 10 and rel_counts[r2] > 10:  # both common enough
                        never_cooccur.append((r1, r2, rel_counts[r1], rel_counts[r2]))
        
        return {
            'top_cooccur': sorted_by_lift[:20],
            'anti_cooccur': sorted_by_lift[-20:],  # lift < 1 means they avoid each other
            'never_together': never_cooccur[:20],
            'relation_counts': dict(rel_counts),
        }
    
    # ============ CROSS VS WITHIN GENERATION ============
    
    def cross_vs_within_generation(self):
        """
        how many relations are vertical (parent-child, grandparent) vs horizontal (sibling, cousin)
        tells us about graph structure
        """
        vertical = []  # cross-generation
        horizontal = []  # same generation
        unknown = []
        
        for h, r, t in self.triplets:
            h_gen = self.features['people'].get(h, {}).get('generation')
            t_gen = self.features['people'].get(t, {}).get('generation')
            
            if h_gen is None or t_gen is None:
                unknown.append((h, r, t))
            elif h_gen == t_gen:
                horizontal.append((h, r, t))
            else:
                vertical.append((h, r, t))
        
        # break down vertical by direction
        downward = [e for e in vertical if self.features['people'][e[0]]['generation'] > self.features['people'][e[2]]['generation']]
        upward = [e for e in vertical if self.features['people'][e[0]]['generation'] < self.features['people'][e[2]]['generation']]
        
        # count relation types in each bucket
        horiz_rels = Counter(r for h, r, t in horizontal)
        vert_rels = Counter(r for h, r, t in vertical)
        
        return {
            'total': len(self.triplets),
            'horizontal': len(horizontal),
            'vertical': len(vertical),
            'unknown': len(unknown),
            'ratio_h_v': round(len(horizontal) / len(vertical), 3) if vertical else None,
            'vertical_downward': len(downward),  # older -> younger
            'vertical_upward': len(upward),  # younger -> older
            'horizontal_relations': dict(horiz_rels),
            'vertical_relations': dict(vert_rels),
        }
    
    # ============ DASHBOARD HELPERS ============
    
    def get_person_summary(self, person_id):
        """
        quick summary for dashboard hover/click
        """
        f = self.features['people'].get(person_id)
        if not f:
            return None
        
        return {
            'id': person_id,
            'generation': f['generation'],
            'gender': 'F' if f['gender_evidence']['female_weight'] > f['gender_evidence']['male_weight'] else 'M' if f['gender_evidence']['male_weight'] > 0 else '?',
            'degree': f['relation_counts']['total'],
            'parents': f['parents']['mothers'] + f['parents']['fathers'],
            'children_count': f['children']['total'],
            'siblings': f['siblings'],
        }
    
    def get_ego_network(self, person_id, hops=1):
        """
        get subgraph around a person for dashboard visualization
        """
        nodes = {person_id}
        edges = []
        
        frontier = {person_id}
        for _ in range(hops):
            new_frontier = set()
            for p in frontier:
                for r, neighbor in self.outgoing[p]:
                    edges.append((p, r, neighbor))
                    new_frontier.add(neighbor)
                for r, neighbor in self.incoming[p]:
                    edges.append((neighbor, r, p))
                    new_frontier.add(neighbor)
            nodes.update(new_frontier)
            frontier = new_frontier
        
        # dedupe edges
        edges = list(set(edges))
        
        return {
            'center': person_id,
            'nodes': list(nodes),
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges),
        }
    
    def get_family_subgraph(self, person_id):
        """
        get nuclear family + grandparents for a person
        useful for family tree view
        """
        f = self.features['people'].get(person_id, {})
        
        nodes = {person_id}
        edges = []
        
        # parents
        parents = f.get('parents', {})
        for m in parents.get('mothers', []):
            nodes.add(m)
            edges.append((m, 'motherOf', person_id))
        for fa in parents.get('fathers', []):
            nodes.add(fa)
            edges.append((fa, 'fatherOf', person_id))
        
        # siblings
        for sib in f.get('siblings', []):
            nodes.add(sib)
            # figure out relation direction
            if (person_id, 'sisterOf', sib) in self.edge_set or (person_id, 'brotherOf', sib) in self.edge_set:
                for r, t in self.outgoing[person_id]:
                    if t == sib and r in SIBLING_RELATIONS:
                        edges.append((person_id, r, sib))
                        break
        
        # children
        for r, child in self.outgoing[person_id]:
            if r in PARENT_RELATIONS:
                nodes.add(child)
                edges.append((person_id, r, child))
        
        # grandparents (parents of parents)
        for parent in parents.get('mothers', []) + parents.get('fathers', []):
            pf = self.features['people'].get(parent, {})
            for gm in pf.get('parents', {}).get('mothers', []):
                nodes.add(gm)
                edges.append((gm, 'motherOf', parent))
            for gf in pf.get('parents', {}).get('fathers', []):
                nodes.add(gf)
                edges.append((gf, 'fatherOf', parent))
        
        return {
            'center': person_id,
            'nodes': list(nodes),
            'edges': list(set(edges)),
        }


def run_exploration(data_path='data/train.txt', output_dir='outputs'):
    """run all explorations and dump results"""
    
    print("loading...")
    loader = MetaFAMLoader(data_path)
    loader.load()
    
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    explorer = Explorer(loader.triplets, features)
    
    results = {}
    
    print("checking derivability...")
    results['derivability'] = explorer.derivability_extended()
    
    print("computing generation-normalized degrees...")
    results['gen_normalized'] = explorer.generation_normalized_degree()
    
    print("sampling path multiplicities...")
    results['path_multiplicity'] = explorer.path_multiplicity_sample(n_pairs=200)
    
    print("computing relation co-occurrence...")
    results['cooccurrence'] = explorer.relation_cooccurrence()
    
    print("analyzing cross vs within generation...")
    results['gen_structure'] = explorer.cross_vs_within_generation()
    
    # save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'exploration_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nsaved to {output_dir}/exploration_results.json")
    
    # print summary
    print("\n" + "="*60)
    print("EXPLORATION SUMMARY")
    print("="*60)
    
    print("\nDerivability:")
    for rel, stats in results['derivability'].items():
        if stats.get('pct') is not None:
            print(f"  {rel}: {stats['pct']}% derivable ({stats['derivable']}/{stats['total']})")
    
    print("\nGeneration-normalized outliers (high degree for their gen):")
    for person, data in results['gen_normalized']['outliers_high'][:5]:
        print(f"  {person}: z={data['z_score']} (raw={data['raw_degree']}, gen {data['generation']} avg={data['gen_mean']})")
    
    print("\nPath multiplicity (sampled pairs):")
    pm = results['path_multiplicity']
    print(f"  connected pairs: {pm['connected_pairs']}/{pm['sampled']}")
    print(f"  avg paths when connected: {pm['avg_paths_when_connected']}")
    print(f"  max paths found: {pm['max_paths']}")
    
    print("\nRelation structure:")
    gs = results['gen_structure']
    print(f"  horizontal (same gen): {gs['horizontal']} ({gs['horizontal']/gs['total']*100:.1f}%)")
    print(f"  vertical (cross gen): {gs['vertical']} ({gs['vertical']/gs['total']*100:.1f}%)")
    print(f"  ratio h/v: {gs['ratio_h_v']}")
    
    print("\nTop relation co-occurrences (by lift):")
    for (r1, r2), data in results['cooccurrence']['top_cooccur'][:5]:
        print(f"  {r1} + {r2}: lift={data['lift']} (count={data['count']})")
    
    return explorer, results


if __name__ == "__main__":
    explorer, results = run_exploration()