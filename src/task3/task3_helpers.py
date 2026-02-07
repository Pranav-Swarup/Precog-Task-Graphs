# only the index builder

import os, sys
from collections import defaultdict
from src.task1.data_loader import MetaFAMLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_and_index(filepath='data/train.txt'):
    
    loader = MetaFAMLoader(filepath)
    loader.load()

    outgoing = defaultdict(lambda: defaultdict(set))  # outgoing[X][r] = {targets}
    triple_set = set()
    relations = set()
    heads_of = defaultdict(set)

    # i need heads_of to check and tell if some guy is appearing in a false or contradictiing relation
    # eg if X is a motherOf Y, then X cannot be a sonOf anyone
    # FUTURE ME EDIT: this is the reason why theres only 750 two step rules mined out of the 28^2 = 784 ones possible.
    # NOTE: ADD TO LATEX DOC

    for h, r, t in loader.triplets:
        outgoing[h][r].add(t)
        triple_set.add((h, r, t))
        relations.add(r)
        heads_of[r].add(h)

    return outgoing, triple_set, sorted(relations), heads_of