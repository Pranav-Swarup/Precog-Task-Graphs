# evaluates stuff based on known relations from common sense on a family KG.

import csv, os, sys
from src.task3.task3_helpers import load_and_index

OUT_DIR = 'raw_csv_outputs/task3'
DATADIR = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'
os.makedirs(OUT_DIR, exist_ok=True)


def inverse(r1, r2, name, outgoing, triple_set):
    
    body = support = 0
    examples, counter = [], []
    
    for X in outgoing:
        if r1 not in outgoing[X]:
            continue
    
        for Y in outgoing[X][r1]:
            body += 1
    
            if (Y, r2, X) in triple_set:
                support += 1
    
                if len(examples) < 3:
                    examples.append(f"{X}-[{r1}]->{Y} => {Y}-[{r2}]->{X}")
    
            else:
                if len(counter) < 2:
                    counter.append(f"{X}-[{r1}]->{Y} but NO {Y}-[{r2}]->{X}")
    
    
    return {'name': name, 'type': 'inverse', 'r1': r1, 'r2': r2, 'r3': '',
            'rule_str': f"{r1}(X,Y) -> {r2}(Y,X)",
            'support': support, 'body': body,
            'confidence': support / body if body > 0 else 0,
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
            'example_3': examples[2] if len(examples) > 2 else '',
            'counter_1': counter[0] if len(counter) > 0 else '',
            'counter_2': counter[1] if len(counter) > 1 else ''}



def two_hop(r1, r2, r3, name, outgoing, triple_set):

    body_pairs, support_pairs = set(), set()
    examples, counter = [], []
    
    for X in outgoing:
        if r1 not in outgoing[X]:
            continue
    
        for Y in outgoing[X][r1]:
            if r2 not in outgoing[Y]:
                continue
    
            for Z in outgoing[Y][r2]:
                if X == Z:
                    continue
                body_pairs.add((X, Z))
    
                if (X, r3, Z) in triple_set:
                    support_pairs.add((X, Z))
    
                    if len(examples) < 3:
                        examples.append(f"{X}-[{r1}]->{Y}-[{r2}]->{Z} => {X}-[{r3}]->{Z}")
                else:
                    if len(counter) < 2:
                        counter.append(f"{X}-[{r1}]->{Y}-[{r2}]->{Z} but NO {X}-[{r3}]->{Z}")
    
    bs, s = len(body_pairs), len(support_pairs)
    
    
    return {'name': name, 'type': 'two_hop', 'r1': r1, 'r2': r2, 'r3': r3,
            'rule_str': f"{r1}(X,Y) ^ {r2}(Y,Z) -> {r3}(X,Z)",
            'support': s, 'body': bs,
            'confidence': s / bs if bs > 0 else 0,
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
            'example_3': examples[2] if len(examples) > 2 else '',
            'counter_1': counter[0] if len(counter) > 0 else '',
            'counter_2': counter[1] if len(counter) > 1 else ''}


def three_hop(r1, r2, r3, r4, name, outgoing, triple_set):
    
    
    body_pairs, support_pairs = set(), set()
    examples = []
    
    for X in outgoing:
    
        if r1 not in outgoing[X]:
            continue
    
        for Y in outgoing[X][r1]:
            if r2 not in outgoing[Y]:
                continue
    
            for Z in outgoing[Y][r2]:
                if r3 not in outgoing[Z]:
                    continue
    
                for W in outgoing[Z][r3]:
                    if X == W:
                        continue
    
                    body_pairs.add((X, W))
    
                    if (X, r4, W) in triple_set:
                        support_pairs.add((X, W))
    
                        if len(examples) < 3:
                            examples.append(f"{X}->{Y}->{Z}->{W} => {X}-[{r4}]->{W}")
    
    
    bs, s = len(body_pairs), len(support_pairs)
    
    
    return {'name': name, 'type': 'three_hop', 'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
            'rule_str': f"{r1}(X,Y) ^ {r2}(Y,Z) ^ {r3}(Z,W) -> {r4}(X,W)",
            'support': s, 'body': bs,
            'confidence': s / bs if bs > 0 else 0,
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
            'example_3': examples[2] if len(examples) > 2 else '',
            'counter_1': '', 'counter_2': ''}


def co_parent(r_parent, r_sib_options, name, outgoing, triple_set):
    
    body_pairs, support_pairs = set(), set()
    examples = []
    
    for X in outgoing:
    
        if r_parent not in outgoing[X]:
            continue
        children = list(outgoing[X][r_parent])
    
        for i in range(len(children)):
    
            for j in range(i + 1, len(children)):
    
                Y, Z = children[i], children[j]
                key = (min(Y, Z), max(Y, Z))
                body_pairs.add(key)
    
                if any((Y, rs, Z) in triple_set or (Z, rs, Y) in triple_set for rs in r_sib_options):
                    support_pairs.add(key)
    
                    if len(examples) < 3:
                        examples.append(f"{r_parent}({X},{Y}) ^ {r_parent}({X},{Z}) -> sibling({Y},{Z})")
    
    
    bs, s = len(body_pairs), len(support_pairs)
    
    
    return {'name': name, 'type': 'co_parent', 'r1': r_parent, 'r2': r_parent, 'r3': 'sibling',
            'rule_str': f"{r_parent}(X,Y) ^ {r_parent}(X,Z) ^ Y!=Z -> sibling(Y,Z)",
            'support': s, 'body': bs,
            'confidence': s / bs if bs > 0 else 0,
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
            'example_3': examples[2] if len(examples) > 2 else '',
            'counter_1': '', 'counter_2': ''}


def run_all(outgoing, triple_set):
    results = []

# THESE RELATIONS WERE FIRST WRITTEN ON PEN AND PAPER AND GIVEN TO CLAUDE TO VERIFY AND EXPAND ON.
# i did not want to miss out on any case to check so these are LLM generated


# CLAUDE GENERATED CODE BEGINS !!!!!!!!!!!!!!!!!!!!!!!

    # Inverse
    for r1, r2, name in [
        ('motherOf', 'daughterOf', 'motherOf -> daughterOf [female child]'),
        ('motherOf', 'sonOf',      'motherOf -> sonOf [male child]'),
        ('fatherOf', 'daughterOf', 'fatherOf -> daughterOf [female child]'),
        ('fatherOf', 'sonOf',      'fatherOf -> sonOf [male child]'),
        ('sisterOf', 'sisterOf',   'sisterOf symmetric [Y female]'),
        ('sisterOf', 'brotherOf',  'sisterOf -> brotherOf [Y male]'),
        ('brotherOf', 'brotherOf', 'brotherOf symmetric [Y male]'),
        ('brotherOf', 'sisterOf',  'brotherOf -> sisterOf [Y female]'),
        ('auntOf', 'nieceOf',      'auntOf -> nieceOf [Y female]'),
        ('auntOf', 'nephewOf',     'auntOf -> nephewOf [Y male]'),
        ('uncleOf', 'nieceOf',     'uncleOf -> nieceOf [Y female]'),
        ('uncleOf', 'nephewOf',    'uncleOf -> nephewOf [Y male]'),
    ]:
        results.append(inverse(r1, r2, name, outgoing, triple_set))

    # Two-hop
    for r1, r2, r3, name in [
        ('motherOf', 'motherOf',   'grandmotherOf', 'grandmother via mother chain'),
        ('motherOf', 'fatherOf',   'grandmotherOf', 'grandmother via mother->father'),
        ('fatherOf', 'motherOf',   'grandfatherOf', 'grandfather via father->mother'),
        ('fatherOf', 'fatherOf',   'grandfatherOf', 'grandfather via father chain'),
        ('sisterOf', 'motherOf',   'auntOf',        'aunt = sister of mother'),
        ('sisterOf', 'fatherOf',   'auntOf',        'aunt = sister of father'),
        ('brotherOf', 'motherOf',  'uncleOf',       'uncle = brother of mother'),
        ('brotherOf', 'fatherOf',  'uncleOf',       'uncle = brother of father'),
        ('daughterOf', 'sisterOf', 'nieceOf',       'niece via daughter->sister'),
        ('daughterOf', 'brotherOf','nieceOf',       'niece via daughter->brother'),
        ('sonOf', 'sisterOf',     'nephewOf',       'nephew via son->sister'),
        ('sonOf', 'brotherOf',    'nephewOf',       'nephew via son->brother'),
        ('motherOf', 'brotherOf', 'motherOf',       'mother of sibling = your mother'),
        ('motherOf', 'sisterOf',  'motherOf',       'mother of sibling = your mother (sis)'),
        ('fatherOf', 'brotherOf', 'fatherOf',       'father of sibling = your father'),
        ('fatherOf', 'sisterOf',  'fatherOf',       'father of sibling = your father (sis)'),
    ]:
        results.append(two_hop(r1, r2, r3, name, outgoing, triple_set))

    # Three-hop
    for r1, r2, r3, r4, name in [
        ('motherOf', 'motherOf', 'motherOf', 'greatGrandmotherOf', 'great-grandmother (m->m->m)'),
        ('fatherOf', 'fatherOf', 'fatherOf', 'greatGrandfatherOf', 'great-grandfather (f->f->f)'),
        ('motherOf', 'motherOf', 'fatherOf', 'greatGrandmotherOf', 'great-grandmother (m->m->f)'),
        ('fatherOf', 'fatherOf', 'motherOf', 'greatGrandfatherOf', 'great-grandfather (f->f->m)'),
    ]:
        results.append(three_hop(r1, r2, r3, r4, name, outgoing, triple_set))


# CLAUDE GENERATED CODE ENDS !!!!!!!!!!!!!!!!!!!!!!!
    
    
    # coparent need to check both ways mom and dad
    results.append(co_parent('motherOf', ['sisterOf', 'brotherOf'], 'shared mother -> siblings', outgoing, triple_set))
    results.append(co_parent('fatherOf', ['sisterOf', 'brotherOf'], 'shared father -> siblings', outgoing, triple_set))

    return results


if __name__ == "__main__":

    filepath = DATADIR

    outgoing, triple_set, relations, heads_of = load_and_index(filepath)

    results = run_all(outgoing, triple_set)

    path = os.path.join(OUT_DIR, 'known_rules_check.csv')

    fields = ['name', 'type', 'rule_str', 'r1', 'r2', 'r3',
              'support', 'body', 'confidence',
              'example_1', 'example_2', 'example_3', 'counter_1', 'counter_2']

    with open(path, 'w', newline='') as f:

        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    print(f"Saved {path} ({len(results)} rules)")