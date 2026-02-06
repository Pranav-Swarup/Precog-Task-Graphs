import networkx as nx # pyright: ignore[reportMissingModuleSource]
from collections import defaultdict
from src.task1.constants import *

# semantic distance: custom defined - lower = closer relationship

RELATION_DISTANCE = {
    'motherOf': 1.0, 'fatherOf': 1.0,
    'daughterOf': 1.0, 'sonOf': 1.0,
    'sisterOf': 0.5, 'brotherOf': 0.5,
    'grandmotherOf': 2.0, 'grandfatherOf': 2.0,
    'granddaughterOf': 2.0, 'grandsonOf': 2.0,
    'greatGrandmotherOf': 3.0, 'greatGrandfatherOf': 3.0,
    'greatGranddaughterOf': 3.0, 'greatGrandsonOf': 3.0,
    'auntOf': 1.5, 'uncleOf': 1.5,
    'nieceOf': 1.5, 'nephewOf': 1.5,
    'greatAuntOf': 2.5, 'greatUncleOf': 2.5,
    'secondAuntOf': 2.0, 'secondUncleOf': 2.0,
    'girlCousinOf': 2.0, 'boyCousinOf': 2.0,
    'girlSecondCousinOf': 3.0, 'boySecondCousinOf': 3.0,
    'girlFirstCousinOnceRemovedOf': 2.5, 'boyFirstCousinOnceRemovedOf': 2.5,
}


def full_undirected(triplets):

    G = nx.Graph()
    for h, r, t in triplets:
        G.add_edge(h, t)
    return G


def nuclear_family_graph(triplets):

    keep = PARENT_RELATIONS | CHILD_RELATIONS | SIBLING_RELATIONS
    G = nx.Graph()
    for h, r, t in triplets:
        if r in keep:
            G.add_edge(h, t)
    return G


def generational_affinity_graph(triplets):
    """
    Same-generation edges get HIGH weight (close affinity),
    cross-generation edges get LOW weight.
    """
    G = nx.Graph()
    for h, r, t in triplets:
        delta = abs(GENERATION_DELTAS.get(r, 1))
        weight = 1.0 / (1.0 + delta)
        if G.has_edge(h, t):
            G[h][t]['weight'] = max(G[h][t]['weight'], weight)
        else:
            G.add_edge(h, t, weight=weight)
    return G


def relation_weighted_graph(triplets):
    """Weighted by semantic closeness. Sibling = high weight, distant = low weight."""
    G = nx.Graph()
    for h, r, t in triplets:
        dist = RELATION_DISTANCE.get(r, 2.0)
        weight = 1.0 / dist
        if G.has_edge(h, t):
            G[h][t]['weight'] = max(G[h][t]['weight'], weight)
        else:
            G.add_edge(h, t, weight=weight)
    return G


def parent_child_dag(triplets):
    """Directed: parent → child only."""
    G = nx.DiGraph()
    for h, r, t in triplets:
        if r in PARENT_RELATIONS:
            G.add_edge(h, t)
        elif r in CHILD_RELATIONS:
            G.add_edge(t, h)
    return G


def get_families(G_undirected):
    """Returns list of node sets, one per connected component."""
    return list(nx.connected_components(G_undirected))


def family_subgraph(G, family_nodes):
    return G.subgraph(family_nodes).copy()


def family_triplets(triplets, family_nodes):
    nodes = set(family_nodes)
    return [(h, r, t) for h, r, t in triplets if h in nodes and t in nodes]