# AMIE style inductive rule mining (check latex doc for citations)

import csv, os, sys
from collections import Counter, defaultdict
from src.task3.task3_helpers import load_and_index

OUT_DIR = 'raw_csv_outputs/task3'
DATADIR = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
os.makedirs(OUT_DIR, exist_ok=True)

MIN_SUPPORT = 5         # tweakable parameter here 


def find_examples(outgoing, triple_set, r1, r2, r3=None, rule_type='two_hop', n=3):

    examples = []
    if rule_type == 'inverse':
        for X in outgoing:

            if r1 not in outgoing[X]:
                continue
            
            for Y in outgoing[X][r1]:
            
                if (Y, r2, X) in triple_set:
                    examples.append(f"{X}-[{r1}]->{Y} => {Y}-[{r2}]->{X}")
            
                    if len(examples) >= n:
                        return examples
    else:
        
        for X in outgoing:
            if r1 not in outgoing[X]:
                continue
        
            for Y in outgoing[X][r1]:
                if r2 not in outgoing[Y]:
                    continue
        
                for Z in outgoing[Y][r2]:
        
                    if X != Z and (X, r3, Z) in triple_set:
                        examples.append(f"{X}-[{r1}]->{Y}-[{r2}]->{Z} => {X}-[{r3}]->{Z}")
                        if len(examples) >= n:
                            return examples
    return examples


def mine_inverse(outgoing, triple_set, relations):
    
    print("Mining inverse rules...")
    rules = []
    
    for r1 in relations:
    
        for r2 in relations:
            support = 0
            body = 0
    
            for X in outgoing:
    
                if r1 not in outgoing[X]:
                    continue
    
                for Y in outgoing[X][r1]:
                    body += 1
                    if (Y, r2, X) in triple_set:
                        support += 1
    
            if support >= MIN_SUPPORT and body > 0:
                rules.append({'r1': r1, 'r2': r2, 'support': support, 'body': body, 'confidence': support / body})

    rules.sort(key=lambda x: (-x['confidence'], -x['support']))
    print(f"  Found {len(rules)} inverse rules")

    # add examples
    for r in rules:
        exs = find_examples(outgoing, triple_set, r['r1'], r['r2'], rule_type='inverse')
    
        for i, ex in enumerate(exs):
            r[f'example_{i+1}'] = ex

# CLAUDE GENERATED CODE FOR CSV WRITING BEGINS !!!!!!!!!!!!!!!!!!!!!

    # write csv
    path = os.path.join(OUT_DIR, 'inverse_rules.csv')
    fields = ['r1', 'r2', 'support', 'body', 'confidence', 'example_1', 'example_2', 'example_3']
    
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rules)
    
    print(f"  Saved {path}")
    return rules


# CLAUDE GENERATED CODE FOR CSV WRITING ENDS !!!!!!!!!!!!!!!!!!!!!

def mine_two_hop(outgoing, triple_set, relations):

    print("Mining two-hop rules")

    rules = []
    
    n_rels = len(relations)
    
    checked = 0

    for r1 in relations:
    
        for r2 in relations:
    
            checked += 1
            if checked % 100 == 0:
                print(f"{checked}/{n_rels**2} pairs added")

            body_pairs = set()
            for X in outgoing:
                if r1 not in outgoing[X]:
                    continue

                for Y in outgoing[X][r1]:
                
                    if r2 not in outgoing[Y]:
                        continue
                
                    for Z in outgoing[Y][r2]:
                        if X != Z:
                            body_pairs.add((X, Z))

            if not body_pairs:
                continue

            conclusion_counts = Counter()
            for (X, Z) in body_pairs:
                
                for r3 in outgoing[X]:
                    if Z in outgoing[X][r3]:
                        conclusion_counts[r3] += 1

            body_size = len(body_pairs)
            for r3, support in conclusion_counts.items():
                if support >= MIN_SUPPORT:
                    rules.append({'r1': r1, 'r2': r2, 'r3': r3, 'support': support, 'body': body_size, 'confidence': support / body_size})

    rules.sort(key=lambda x: (-x['confidence'], -x['support']))
    print(f"  Found {len(rules)} two-hop rules")

    # add examples
    for r in rules:
        exs = find_examples(outgoing, triple_set, r['r1'], r['r2'], r['r3'], rule_type='two_hop')
        for i, ex in enumerate(exs):
            r[f'example_{i+1}'] = ex

    path = os.path.join(OUT_DIR, 'two_hop_rules.csv')
    fields = ['r1', 'r2', 'r3', 'support', 'body', 'confidence', 'example_1', 'example_2', 'example_3']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rules)
    print(f"  Saved {path}")
    return rules


if __name__ == "__main__":

    filepath = DATADIR
    outgoing, triple_set, relations, heads_of = load_and_index(filepath)
    mine_inverse(outgoing, triple_set, relations)
    mine_two_hop(outgoing, triple_set, relations)
    print("Amie Rule mining done")