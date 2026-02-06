# THIS FILE IS HERE TO ONLY GET THE RAW STUFF FROM THE DATASET

# REMINDER - DO NOT MAKE ANY INFERENCES IN THIS FILE DO THAT IN PRIMARY OR SEC ANALYSIS

from collections import defaultdict, Counter
from typing import *
from src.task1.constants import *

class RawFeatureExtractor:
    
    def __init__(self, triplets, people):
        self.triplets = triplets
        self.people = people
        
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

        # i a m rebuilding the triplets here 

        for head, rel, tail in triplets:
            self.outgoing[head].append((rel, tail))
            self.incoming[tail].append((rel, head))
    
    #  gender
    
    def get_gender_evidence(self, person_id: str) -> Dict:
 
 
        female_weight = 0
        male_weight = 0
        female_sources = []  # which relations contributed
        male_sources = []
        
        for rel, target in self.outgoing[person_id]:
            if rel in GENDER_EVIDENCE:
                gender, weight = GENDER_EVIDENCE[rel]
                if gender == 'F':
                    female_weight += weight
                    female_sources.append(rel)
                else:
                    male_weight += weight
                    male_sources.append(rel)
        
        return {
            'female_weight': female_weight,
            'male_weight': male_weight,
            'female_sources': female_sources,
            'male_sources': male_sources,
            'total_weight': female_weight + male_weight,
            # not makin the call here, just providing the ratio
            'female_ratio': female_weight / (female_weight + male_weight) if (female_weight + male_weight) > 0 else None,
        }
    

    
    def propagate_generations(self) -> Dict:


        # build constraint edges
        neighbors = defaultdict(list)

        for head, rel, tail in self.triplets:
        
            if rel in GENERATION_DELTAS:
        
                delta = GENERATION_DELTAS[rel]
                neighbors[head].append((tail, delta))
                neighbors[tail].append((head, -delta))
        
        visited = set()
        generations = {}

        # store ALL gen values we see for each person, not just first
        
        all_gen_values = defaultdict(list)
        
        for start_person in self.people:
            if start_person in visited:
                continue
            
            queue = [(start_person, 0)]
            component_gens = {}
            
            while queue:
                person, gen = queue.pop(0)
                all_gen_values[person].append(gen)
                
                if person in component_gens:
                    continue
                
                component_gens[person] = gen
                visited.add(person)
                
                for neighbor, delta in neighbors[person]:
                    neighbor_gen = gen - delta
                    queue.append((neighbor, neighbor_gen))
            
            # normalize so min is 0
            if component_gens:
                min_gen = min(component_gens.values())
                for p in component_gens:
                    generations[p] = component_gens[p] - min_gen
        
        # flag people with no generation informative relations
        for person in self.people:
            if person not in generations:
                generations[person] = None
        
        return {
            'generations': generations,
            'all_values_seen': dict(all_gen_values),  # for conflict analysis
        }
    
    # family structure functions
    
    def get_parents(self, person_id: str) -> Dict:
        parents = []  # cant always tell mother vs father from daughterOf/sonOf
        mothers = []
        fathers = []
        
        # method 1 incoming motherOf/fatherOf
        # "Y motherOf X" means Y is X's mother
        for rel, source in self.incoming[person_id]:
            
            if rel == 'motherOf':
                mothers.append(source)
            elif rel == 'fatherOf':
                fathers.append(source)
        
        # method 2outgoing daughterOf/sonOf
        # "X daughterOf Y" means Y is X's parent gender unknown from this alone

        for rel, target in self.outgoing[person_id]:
            if rel in ('daughterOf', 'sonOf'):
                # check if we already have this person from method 1
                if target not in mothers and target not in fathers:
                    parents.append(target)
        
        return {
            'mothers': mothers,
            'fathers': fathers,
            'unknown_gender_parents': parents,  # from daughterOf/sonOf
            'total': len(mothers) + len(fathers) + len(parents),
        }
    

    
    def get_children(self, person_id: str) -> Dict:
        
        children = []
        
        #  outgoing motherOf/fatherOf
        # X motherOf y means Y is X's child
        for rel, target in self.outgoing[person_id]:
            if rel in PARENT_RELATIONS:
                children.append(target)
        
        #  incoming daughterOf/sonOf
        # Y daughterOf X means Y is X's child
        for rel, source in self.incoming[person_id]:
            if rel in ('daughterOf', 'sonOf'):
                if source not in children:
                    children.append(source)
        
        return {
            'children': children,
            'total': len(children),
        }

    """  
    # deprecated dont use this quick gender funtion !!!!

    def _quick_gender_check(self, person_id: str) -> str:
        
        for rel, smtg in self.outgoing[person_id]:
            
            if rel in GENDER_EVIDENCE:
                return GENDER_EVIDENCE[rel][0]
        
        return 'U'
    
    """
    
    def get_siblings(self, person_id: str) -> List[str]:
        
        sibs = set()

        for rel, target in self.outgoing[person_id]:
            if rel in SIBLING_RELATIONS:
                sibs.add(target)
        
        for rel, source in self.incoming[person_id]:
            if rel in SIBLING_RELATIONS:
                sibs.add(source)
        
        return list(sibs)
    
    
    def count_relations(self, person_id: str) -> Dict:
        
        out_counts = Counter()
        in_counts = Counter()
        
        for rel, adwadad in self.outgoing[person_id]:
            out_counts[rel] += 1
        
        for rel, awdawda in self.incoming[person_id]:
            in_counts[rel] += 1
        
        return {
        
            'outgoing': dict(out_counts),
            'incoming': dict(in_counts),
            'total_out': sum(out_counts.values()),
            'total_in': sum(in_counts.values()),
            'total': sum(out_counts.values()) + sum(in_counts.values()),
        
        }
    

    def detect_anomalies(self, person_id: str, gen_data: Dict) -> List[Dict]:
        
        # finds anything weird, each record has type, details, severity for filtering later
        anomalies = []
        
        # gender
        gender_ev = self.get_gender_evidence(person_id)


        if gender_ev['female_weight'] > 0 and gender_ev['male_weight'] > 0:

            anomalies.append({

                'type': 'gender_mixed_evidence',
                'female_weight': gender_ev['female_weight'],
                'male_weight': gender_ev['male_weight'],
                'female_sources': gender_ev['female_sources'],
                'male_sources': gender_ev['male_sources'],

                # severity based on how close the weights are
                'severity': min(gender_ev['female_weight'], gender_ev['male_weight']) / max(gender_ev['female_weight'], gender_ev['male_weight']),
            })
        
        # parents
        parents = self.get_parents(person_id)


        if len(parents['mothers']) > 1:
            anomalies.append({
                'type': 'multiple_mothers',
                'count': len(parents['mothers']),
                'mothers': parents['mothers'],
                'severity': 1.0,              
                # this is definitely weird unless u consider stepmoms like the Dudleys
            })


        if len(parents['fathers']) > 1:

            anomalies.append({
                'type': 'multiple_fathers',
                'count': len(parents['fathers']),
                'fathers': parents['fathers'],
                'severity': 1.0,
            })


        if parents['total'] > 2:
            
            anomalies.append({
                'type': 'too_many_parents',
                'count': parents['total'],
                'severity': 1.0,
            })
        
        # generation conflicts
        if person_id in gen_data['all_values_seen']:

            gen_vals = gen_data['all_values_seen'][person_id]
            unique_vals = set(gen_vals)
            
            if len(unique_vals) > 1:
                anomalies.append({
                    'type': 'generation_conflict',
                    'values_seen': list(unique_vals),
                    'assigned': gen_data['generations'].get(person_id),
                    'severity': max(unique_vals) - min(unique_vals),  # how big is the spread
                })
        
        # self loops this is bad if there are any
        for rel, target in self.outgoing[person_id]:
            
            if target == person_id:
                anomalies.append({
                    'type': 'self_relation',
                    'relation': rel,
                    'severity': 1.0,
                })
        
        return anomalies
    
    # main extraction function
    
    def extract_all(self) -> Dict:

        gen_data = self.propagate_generations()
        
        features = {}
        
        for person_id in self.people:
            features[person_id] = {


                'person_id': person_id,
                'gender_evidence': self.get_gender_evidence(person_id),
                'generation': gen_data['generations'].get(person_id),

                'parents': self.get_parents(person_id),
                'children': self.get_children(person_id),
                'siblings': self.get_siblings(person_id),
                
                'relation_counts': self.count_relations(person_id),
                
                'anomalies': self.detect_anomalies(person_id, gen_data),
            }
        
        return {
            'people': features,
            'generation_data': gen_data,
        }
