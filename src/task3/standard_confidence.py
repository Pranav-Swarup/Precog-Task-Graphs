# in hindsight this was a naive approach to confidence
# here i treat all missing values are erroneous but they just indicate that theres relations missing.
# pca is better suited.

import csv, os, sys
from src.task3.task3_helpers import load_and_index

OUT_DIR = 'raw_csv_outputs/task3'
DATADIR = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'

def standard_confidence_two_hop(r1, r2, r3, outgoing, triple_set):
    
    body_pairs = set()
    support = 0
    
    for X in outgoing:
        if r1 not in outgoing[X]:
            continue
    
        for Y in outgoing[X][r1]:
            if r2 not in outgoing[Y]:
                continue
    
            for Z in outgoing[Y][r2]:
                if X != Z:
                    body_pairs.add((X, Z))
                    if (X, r3, Z) in triple_set:
                        support += 1
    
    return support, len(body_pairs), support / len(body_pairs) if body_pairs else 0


def standard_confidence_inverse(r1, r2, outgoing, triple_set):
    
    body = support = 0
    
    for X in outgoing:
        if r1 not in outgoing[X]:
            continue
        for Y in outgoing[X][r1]:
            body += 1
            if (Y, r2, X) in triple_set:
                support += 1
    return support, body, support / body if body > 0 else 0



# CLAUDE GENERATED CODE FOR CSV WRITING BEGINS !!!!!!!!!!!!!!!!!!!!!

if __name__ == "__main__":
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
    outgoing, triple_set, relations, heads_of = load_and_index(filepath)

    rows = []

    
    inv_path = os.path.join(OUT_DIR, 'inverse_rules.csv')
    
    if os.path.exists(inv_path):
    
        with open(inv_path) as f:
            for rule in csv.DictReader(f):
                s, b, c = standard_confidence_inverse(rule['r1'], rule['r2'], outgoing, triple_set)
                rows.append({'source': 'inductive_inverse', 'rule': f"{rule['r1']}(X,Y) -> {rule['r2']}(Y,X)",
                             'r1': rule['r1'], 'r2': rule['r2'], 'r3': '',
                             'support': s, 'body': b, 'std_confidence': round(c, 6)})

    # process inductive two-hop rules
    th_path = os.path.join(OUT_DIR, 'two_hop_rules.csv')
    if os.path.exists(th_path):
        with open(th_path) as f:
            for rule in csv.DictReader(f):
                s, b, c = standard_confidence_two_hop(rule['r1'], rule['r2'], rule['r3'], outgoing, triple_set)
                rows.append({'source': 'inductive_two_hop', 'rule': f"{rule['r1']}(X,Y) ^ {rule['r2']}(Y,Z) -> {rule['r3']}(X,Z)",
                             'r1': rule['r1'], 'r2': rule['r2'], 'r3': rule['r3'],
                             'support': s, 'body': b, 'std_confidence': round(c, 6)})

    # process curated rules (two_hop ones only — inverse already covered)
    cur_path = os.path.join(OUT_DIR, 'known_rules_check.csv')
    
    if os.path.exists(cur_path):
    
        with open(cur_path) as f:
    
            for rule in csv.DictReader(f):
    
                if rule['type'] == 'two_hop':
                    s, b, c = standard_confidence_two_hop(rule['r1'], rule['r2'], rule['r3'], outgoing, triple_set)
                    rows.append({'source': 'curated_two_hop', 'rule': rule['rule_str'],
                                 'r1': rule['r1'], 'r2': rule['r2'], 'r3': rule['r3'],
                                 'support': s, 'body': b, 'std_confidence': round(c, 6)})
    
                elif rule['type'] == 'inverse':
                    s, b, c = standard_confidence_inverse(rule['r1'], rule['r2'], outgoing, triple_set)
                    rows.append({'source': 'curated_inverse', 'rule': rule['rule_str'],
                                 'r1': rule['r1'], 'r2': rule['r2'], 'r3': '',
                                 'support': s, 'body': b, 'std_confidence': round(c, 6)})

    path = os.path.join(OUT_DIR, 'standard_confidence.csv')
    fields = ['source', 'rule', 'r1', 'r2', 'r3', 'support', 'body', 'std_confidence']
    
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    
    print(f"Saved {path} ({len(rows)} rules)")


# CLAUDE GENERATED CODE FOR CSV WRITING ENDS !!!!!!!!!!!!!!!!!!!!!