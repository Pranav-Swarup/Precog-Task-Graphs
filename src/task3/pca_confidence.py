# Partial Completeness Assumption confidence. 
# this is more appropriate for open-world KGs where missing != false.


import csv, os, sys
from src.task3.task3_helpers import load_and_index

OUT_DIR = 'raw_csv_outputs/task3'


def pca_confidence_two_hop(r1, r2, r3, outgoing, triple_set, heads_of):
    
    # pca confidence for two-hop rule.
    # Denominator only counts (X,Z) and penalizes that shi where X is known to participate in r3. this is why i needed heads_of
    # where X already has at least one r3 edge to some entity.

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

    r3_heads = heads_of.get(r3, set())
    pca_body = 0
    pca_support = 0
    for (X, Z) in body_pairs:
        if X in r3_heads:
            pca_body += 1
            if (X, r3, Z) in triple_set:
                pca_support += 1

    # also compute standard for comparison
    std_support = sum(1 for (X, Z) in body_pairs if (X, r3, Z) in triple_set)
    std_body = len(body_pairs)

    return {
        'std_support': std_support, 'std_body': std_body,
        'std_confidence': std_support / std_body if std_body > 0 else 0,
        'pca_support': pca_support, 'pca_body': pca_body,
        'pca_confidence': pca_support / pca_body if pca_body > 0 else 0,
    }


def pca_confidence_inverse(r1, r2, outgoing, triple_set, heads_of):
    
    body = support = 0
    pca_body = pca_support = 0
    r2_heads = heads_of.get(r2, set())

    for X in outgoing:
        if r1 not in outgoing[X]:
            continue
        for Y in outgoing[X][r1]:
            body += 1
            hit = (Y, r2, X) in triple_set
            if hit:
                support += 1
            if Y in r2_heads:
                pca_body += 1
                if hit:
                    pca_support += 1

    return {
        'std_support': support, 'std_body': body,
        'std_confidence': support / body if body > 0 else 0,
        'pca_support': pca_support, 'pca_body': pca_body,
        'pca_confidence': pca_support / pca_body if pca_body > 0 else 0,
    }


# CLAUDE GENERATED CODE FOR CSV WRITING BEGINS !!!!!!!!!!!!!!!!!!!!!

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
    outgoing, triple_set, relations, heads_of = load_and_index(filepath)

    rows = []

    # process inductive inverse
    inv_path = os.path.join(OUT_DIR, 'inverse_rules.csv')
    if os.path.exists(inv_path):
        with open(inv_path) as f:
            for rule in csv.DictReader(f):
                m = pca_confidence_inverse(rule['r1'], rule['r2'], outgoing, triple_set, heads_of)
                rows.append({'source': 'inductive_inverse',
                             'rule': f"{rule['r1']}(X,Y) -> {rule['r2']}(Y,X)",
                             'r1': rule['r1'], 'r2': rule['r2'], 'r3': '', **m})

    # process inductive two-hop
    th_path = os.path.join(OUT_DIR, 'two_hop_rules.csv')
    if os.path.exists(th_path):
        with open(th_path) as f:
            for rule in csv.DictReader(f):
                m = pca_confidence_two_hop(rule['r1'], rule['r2'], rule['r3'], outgoing, triple_set, heads_of)
                rows.append({'source': 'inductive_two_hop',
                             'rule': f"{rule['r1']}(X,Y) ^ {rule['r2']}(Y,Z) -> {rule['r3']}(X,Z)",
                             'r1': rule['r1'], 'r2': rule['r2'], 'r3': rule['r3'], **m})

    # process curated
    cur_path = os.path.join(OUT_DIR, 'curated_rules.csv')
    if os.path.exists(cur_path):
        with open(cur_path) as f:
            for rule in csv.DictReader(f):
                if rule['type'] == 'two_hop':
                    m = pca_confidence_two_hop(rule['r1'], rule['r2'], rule['r3'], outgoing, triple_set, heads_of)
                    rows.append({'source': 'curated_two_hop', 'rule': rule['rule_str'],
                                 'r1': rule['r1'], 'r2': rule['r2'], 'r3': rule['r3'], **m})
                elif rule['type'] == 'inverse':
                    m = pca_confidence_inverse(rule['r1'], rule['r2'], outgoing, triple_set, heads_of)
                    rows.append({'source': 'curated_inverse', 'rule': rule['rule_str'],
                                 'r1': rule['r1'], 'r2': rule['r2'], 'r3': '', **m})

    path = os.path.join(OUT_DIR, 'pca_confidence.csv')
    fields = ['source', 'rule', 'r1', 'r2', 'r3',
              'std_support', 'std_body', 'std_confidence',
              'pca_support', 'pca_body', 'pca_confidence']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path} ({len(rows)} rules)")
    
    # quick summary of where PCA differs from standard
    diffs = [(r['rule'], r['std_confidence'], r['pca_confidence']) 
             for r in rows if abs(r['std_confidence'] - r['pca_confidence']) > 0.01]
    if diffs:
        print(f"\nRules where PCA differs from standard by >1%: {len(diffs)}")
        for rule, std, pca in sorted(diffs, key=lambda x: abs(x[2]-x[1]), reverse=True)[:15]:
            print(f"  {rule[:60]:<60}  std={std:.4f}  pca={pca:.4f}  delta={pca-std:+.4f}")

# CLAUDE GENERATED CODE FOR PCA CONFIDENCE CALCULATION ENDS !!!!!!!!!!!!!!!!!!!!!