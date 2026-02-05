# graph-wide analysis and pattern detection, ts looks at multiple people at once and make claimns

from collections import defaultdict, Counter
from src.constants import SYMMETRIC_GROUPS, DERIVABLE_PATTERNS, PARENT_RELATIONS


class GraphAnalyzer:
    
    def __init__(self, triplets, features):
        self.triplets = triplets
        self.features = features
        self.edge_set = set((h, r, t) for h, r, t in triplets)
        
        # build some indexes
        self.outgoing = defaultdict(list)
        for h, r, t in triplets:
            self.outgoing[h].append((r, t))
    
    def check_symmetry_violations(self) -> list:

        #finds cases where symmetric relation is missing
        #e .g. A sisterOf B but B has no sibling relation to A
 
        violations = []
        
        for head, rel, tail in self.triplets:
            if rel in SYMMETRIC_GROUPS:
                valid_reverses = SYMMETRIC_GROUPS[rel]
                has_reverse = any((tail, vr, head) in self.edge_set for vr in valid_reverses)
                if not has_reverse:
                    violations.append({
                        'head': head,
                        'relation': rel,
                        'tail': tail,
                        'expected_reverse': valid_reverses,
                    })
        
        return violations
    
    # THIS FUNCTION WAS A PAIN TO WRITE BUT NOW IT'S DONE :D

    def check_derivable_relations(self) -> dict:

        # for relations like grandmotherOf, check if the intermeiate parent relations exist
        # returns  how many are derivable vs standalone

        results = {rel: {'total': 0, 'derivable': 0, 'standalone': []} for rel in DERIVABLE_PATTERNS}
        
        for head, rel, tail in self.triplets:
            if rel not in DERIVABLE_PATTERNS:
                continue
            
            results[rel]['total'] += 1
            found_chain = False
            
            for r1, r2 in DERIVABLE_PATTERNS[rel]:

                # check head -[r1]-> mid -[r2]-> tail

                for r, mid in self.outgoing[head]:
                    if r == r1:
                        for r2_check, target in self.outgoing[mid]:
                            if r2_check == r2 and target == tail:
                                found_chain = True
                                break
                    if found_chain:
                        break
                if found_chain:
                    break
            
            if found_chain:
                results[rel]['derivable'] += 1
            else:
                results[rel]['standalone'].append((head, tail))
        
        return results
    
    def detect_nuclear_families(self) -> list:

        # finds (mother, father, [children]) units

        # map child -> parents
        child_to_parents = defaultdict(dict)
        for h, r, t in self.triplets:
            if r == 'motherOf':
                child_to_parents[t]['mother'] = h
            elif r == 'fatherOf':
                child_to_parents[t]['father'] = h
        
        # group by parent pair
        parent_pair_to_children = defaultdict(set)
        for child, parents in child_to_parents.items():
            if 'mother' in parents and 'father' in parents:
                pair = (parents['mother'], parents['father'])
                parent_pair_to_children[pair].add(child)
        
        families = []
        for (mother, father), children in parent_pair_to_children.items():
            families.append({
                'mother': mother,
                'father': father,
                'children': list(children),
                'size': len(children) + 2,
            })
        
        return families
    
    def find_sibling_groups(self) -> list:
        # finds groups of siblings based on shared parents
        # child -> set of parents
        child_parents = defaultdict(set)
        for h, r, t in self.triplets:
            if r in PARENT_RELATIONS:
                child_parents[t].add(h)
        
        # group children by their parent set
        parent_set_to_children = defaultdict(set)
        for child, parents in child_parents.items():
            if len(parents) == 2:  # only full siblings for now
                key = frozenset(parents)
                parent_set_to_children[key].add(child)
        
        groups = []
        for parents, children in parent_set_to_children.items():
            if len(children) > 1:
                groups.append({
                    'parents': list(parents),
                    'siblings': list(children),
                    'size': len(children),
                })
        
        return groups
    
    def compute_generation_stats(self) -> dict:

        gens = [f['generation'] for f in self.features['people'].values() if f['generation'] is not None]
        
        if not gens:
            return {'error': 'no generation data'}
        
        gen_counts = Counter(gens)
        
        return {
            'distribution': dict(gen_counts),
            'min': min(gens),
            'max': max(gens),
            'depth': max(gens) - min(gens) + 1,
            'most_common': gen_counts.most_common(1)[0],
            'least_common': gen_counts.most_common()[-1],
        }
    
    def compute_degree_distribution(self) -> dict:
        
        
        
        degrees = [f['relation_counts']['total'] for f in self.features['people'].values()]
        
        return {
            'min': min(degrees),
            'max': max(degrees),
            'avg': sum(degrees) / len(degrees),
            'distribution': dict(Counter(degrees)),
        }
    
    def find_high_degree_nodes(self, threshold: int = 30) -> list:

        # people with lots of connections might be central figures or data anomalies 
        # like gengis khan or smn

        high_deg = []
        for person_id, f in self.features['people'].items():
            deg = f['relation_counts']['total']
            if deg >= threshold:
                high_deg.append({
                    'person': person_id,
                    'degree': deg,
                    'generation': f['generation'],
                })
        
        return sorted(high_deg, key=lambda x: x['degree'], reverse=True)
    
    def find_isolated_nodes(self) -> list:
        
        # people with very few connections
        # aka me on linkedin. 
        
        isolated = []
        for person_id, f in self.features['people'].items():
            deg = f['relation_counts']['total']
            if deg <= 2:
                isolated.append({
                    'person': person_id,
                    'degree': deg,
                    'relations': f['relation_counts'],
                })
        
        return isolated
