# family_centrality.py these are some rudimentary implementations of domain-aware metrics I tried coming up with
# specifically for family knowledge graphs. based on semantic sense.

import networkx as nx
from collections import defaultdict, Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MetaFAMLoader
from src.feature_extractor import RawFeatureExtractor
from src.inference import infer_gender
from src.constants import PARENT_RELATIONS, SIBLING_RELATIONS


class FamilyCentrality:
    
    
    def __init__(self, triplets, features):
    
        self.triplets = triplets
        self.features = features
        self.people = list(features['people'].keys())
        
        # build parent-child DAG (directed, no sibling edges)
        # this is the "true" genealogical structure
    
        self.parent_child_dag = nx.DiGraph()
        for h, r, t in triplets:
            if r in PARENT_RELATIONS:
                # h is parent of t
                self.parent_child_dag.add_edge(h, t)
        
        # also track reverse for ancestor queries
        self.child_parent_dag = self.parent_child_dag.reverse()
        
        # cache ancestors/descendants  cause its expensive to recompute)
        self._ancestors_cache = {}
        self._descendants_cache = {}
        
        
        self.founders = set()
        for person in self.people:
            if self.parent_child_dag.in_degree(person) == 0 and person in self.parent_child_dag:
                self.founders.add(person)
            elif person not in self.parent_child_dag:
                pass
    
    def _get_ancestors(self, person):
        
        
        if person in self._ancestors_cache:
            return self._ancestors_cache[person]
        
        if person not in self.child_parent_dag:
            self._ancestors_cache[person] = set()
            return set()
        
        ancestors = set(nx.ancestors(self.child_parent_dag, person))
        self._ancestors_cache[person] = ancestors
        return ancestors
    
    def _get_descendants(self, person):
        
        if person in self._descendants_cache:
            return self._descendants_cache[person]
        
        if person not in self.parent_child_dag:
            self._descendants_cache[person] = set()
            return set()
        
        descendants = set(nx.descendants(self.parent_child_dag, person))
        self._descendants_cache[person] = descendants
        return descendants
    
    
    def descendant_counts(self):
    
        drc = {}
        wdrc = {}
        
        for person in self.people:
            descendants = self._get_descendants(person)
            drc[person] = len(descendants)
            
            # weighted version
            person_gen = self.features['people'][person]['generation']
            if person_gen is None:
                wdrc[person] = 0
                continue
            
            weighted_sum = 0
            for d in descendants:
                d_gen = self.features['people'][d]['generation']
                if d_gen is not None and d_gen > person_gen:
                    weighted_sum += 1.0 / (d_gen - person_gen)
            wdrc[person] = weighted_sum
        
        return {'raw': drc, 'weighted': wdrc}
    
    def upward_diversity(self):
        
        ud = {}
        
        for person in self.people:
            ancestors = self._get_ancestors(person)
            # which founders are among ancestors?
            founder_ancestors = ancestors & self.founders
            
            # also check if person is founder themselves
            if person in self.founders:
                ud[person] = 1  # they are their own lineage
            else:
                ud[person] = len(founder_ancestors) if founder_ancestors else 0
        
        return ud
    
    def balance_index(self):
        
        bi = {}
        
        for person in self.people:
            ancestors = self._get_ancestors(person)
            descendants = self._get_descendants(person)
            
            n_anc = len(ancestors)
            n_desc = len(descendants)
            
            bi[person] = (n_desc - n_anc) / (n_desc + n_anc + 1)
        
        return bi
    
    
    
    def lineage_criticality_score(self):
        """
        LCS(v) = number of descendants who would lose ALL ancestor paths if v removed
        
        different from articulation points:
        - articulation points just check graph connectivity
        - LCS checks genealogical continuity to founders
        
        a person with high LCS is a "genealogical bottleneck"
        
        COMPUTATION: for each descendant, check if all paths to founders go through v
        equivalent to: descendant's ancestors ∩ founders ⊆ ancestors reachable via v
        
        SIMPLIFICATION: in practice, if v is the only parent of a child,
        that child (and all their descendants) depend entirely on v for ancestry
        """
        lcs = {}
        
        for person in self.people:
            descendants = self._get_descendants(person)
            person_ancestors = self._get_ancestors(person)
            person_founders = person_ancestors & self.founders
            
            critical_count = 0
            
            for desc in descendants:
                desc_ancestors = self._get_ancestors(desc)
                desc_founders = desc_ancestors & self.founders
    
                founders_not_through_person = desc_founders - person_founders
                
                # also need to account for person themselves if they're a founder
                if person in self.founders:
                    founders_not_through_person.discard(person)
                
                if len(founders_not_through_person) == 0:
                    # all of desc's founder connections go through person
                    critical_count += 1
            
            lcs[person] = critical_count
        
        return lcs
    
    def generation_span(self):
    
        gs = {}
        
        for person in self.people:
            person_gen = self.features['people'][person]['generation']
            if person_gen is None:
                gs[person] = 0
                continue
            
            ancestors = self._get_ancestors(person)
            descendants = self._get_descendants(person)
            
            anc_gens = [self.features['people'][a]['generation'] for a in ancestors 
                       if self.features['people'][a]['generation'] is not None]
            desc_gens = [self.features['people'][d]['generation'] for d in descendants
                        if self.features['people'][d]['generation'] is not None]
            
            min_gen = min(anc_gens) if anc_gens else person_gen
            max_gen = max(desc_gens) if desc_gens else person_gen
            
            gs[person] = max_gen - min_gen
        
        return gs
    
    
    def compute_all(self):
    
        drc = self.descendant_counts()
        
        print("computing ancestral diversity score...")
        ads = self.upward_diversity()
        
        print("computing generational balance index...")
        gbi = self.balance_index()
        
        print("computing lineage criticality score...")
        lcs = self.lineage_criticality_score()
    
        print("computing generation span...")
        gs = self.generation_span()
        
        return {
            'descendant_counts': drc,
            'ancestral_diversity': ads,
            'generational_balance': gbi,
            
            'lineage_criticality': lcs,
            
            'generation_span': gs,
        }
    
    def print_insights(self, results):
        
        
        drc_raw = results['descendant_counts']['raw']
        ads = results['ancestral_diversity']
        gbi = results['generational_balance']
        lcs = results['lineage_criticality']
        gs = results['generation_span']
        
        
        # --- DRC ---
        print("\nDESCENDANT COUNTS")
        print("measures how many people exist downstream from this person")
        
        top_drc = sorted(drc_raw.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("highest descendant counts:")
        for person, score in top_drc:
            gen = self.features['people'][person]['generation']
            print(f"  {person}: {score} descendants (gen {gen})")
        
        # drc by generation
        gen_drc = defaultdict(list)
        for person, score in drc_raw.items():
            gen = self.features['people'][person]['generation']
            if gen is not None:
                gen_drc[gen].append(score)
        
        print("\naverage descendant counts by generation:")
        for g in sorted(gen_drc.keys()):
            avg = sum(gen_drc[g]) / len(gen_drc[g])
            print(f"  gen {g}: {avg:.1f} avg descendants")
            
        
        # --- ADS ---
        print("\nANCESTRAL DIVERSITY SCORE")
        print("measures: how many distinct founder lineages converge at this person")
                
        top_ads = sorted(ads.items(), key=lambda x: x[1], reverse=True)[:10]
        print("highest ancestral diversity:")
        for person, score in top_ads:
            gen = self.features['people'][person]['generation']
            print(f"  {person}: {score} founder lineages (gen {gen})")
        
        # ads by generation
        gen_ads = defaultdict(list)
        for person, score in ads.items():
            gen = self.features['people'][person]['generation']
            if gen is not None:
                gen_ads[gen].append(score)
        
        print("\naverage ancestral diversity by generation:")
        for g in sorted(gen_ads.keys()):
            avg = sum(gen_ads[g]) / len(gen_ads[g])
            print(f"  gen {g}: {avg:.1f} founder lineages")
        
        
        #  BI 
        print("\nBALANCE INDEX")
        print("measures: ratio of descendants to ancestors")
        print("range: [-1 (all ancestors), 0 (balanced), +1 (all descendants)]\n")
        
        # group by GBI range
        founder_like = [(p,g) for p,g in gbi.items() if g > 0.5]
        balanced = [(p,g) for p,g in gbi.items() if -0.3 <= g <= 0.3]
        leaf_like = [(p,g) for p,g in gbi.items() if g < -0.5]
        
        print(f"distribution:")
        print(f"  founder-like (GBI > 0.5): {len(founder_like)} people")
        print(f"  balanced (-0.3 to 0.3): {len(balanced)} people")
        print(f"  leaf-like (GBI < -0.5): {len(leaf_like)} people")
        
        # --- LCS ---
        print("\nLINEAGE CRITICALITY")
        print("measures: descendants who would lose all founder connections if this person removed")
        
        top_lcs = sorted(lcs.items(), key=lambda x: x[1], reverse=True)[:10]
        print("highest lineage criticality:")
        for person, score in top_lcs:
            gen = self.features['people'][person]['generation']
            desc_count = len(self._get_descendants(person))
            print(f"  {person}: {score}/{desc_count} critical descendants (gen {gen})")
        
        # how many people have LCS > 0?
        critical_people = [p for p, s in lcs.items() if s > 0]
        print(f"\npeople with non-zero criticality: {len(critical_people)}")
        
        # --- GS ---
        print("\nGENERATION SPAN")
        print("measures: range of generations covered by person's lineage")
        
        top_gs = sorted(gs.items(), key=lambda x: x[1], reverse=True)[:10]
        print("highest generation span:")
        for person, score in top_gs:
            gen = self.features['people'][person]['generation']
            print(f"  {person}: spans {score} generations (own gen {gen})")
        
        

def run_family_centrality(data_path='data/train.txt'):
    
    print("loading data...")
    loader = MetaFAMLoader(data_path)
    loader.load()
    
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    print(f"computing family-specific centrality for {len(loader.people)} people...")
    fc = FamilyCentrality(loader.triplets, features)
    
    results = fc.compute_all()
    fc.print_insights(results)
    
    return fc, results


if __name__ == "__main__":
    fc, results = run_family_centrality()