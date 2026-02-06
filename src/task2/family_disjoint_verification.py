from src.task1.data_loader import MetaFAMLoader
from collections import deque
import networkx as nx # type: ignore

def manual_bfs_components(G):
    visited = set()
    components = []

    for start in G.nodes():
        if start in visited:
            continue

        component = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            for nbr in G.neighbors(node):
                if nbr not in visited:
                    queue.append(nbr)

        components.append(component)

    return components

def build_graph_from_loader(loader):

    G = nx.Graph()

    for h, r, t in loader.triplets:
        G.add_edge(h, t)

    return G


def main():

    loader = MetaFAMLoader("data/train.txt")
    loader.load()

    G = build_graph_from_loader(loader)

    print(f"\nDataset: {G.number_of_nodes()} nodes, {G.number_of_edges()} undirected edges")


    manual_components = manual_bfs_components(G)
    manual_components = sorted(manual_components, key=len, reverse=True)

    nx_components = list(nx.connected_components(G))
    nx_components = sorted(nx_components, key=len, reverse=True)

    print(f"Manual components: {len(manual_components)}")
    print(f"NetworkX components: {len(nx_components)}")

    manual_set = {frozenset(c) for c in manual_components}
    nx_set = {frozenset(c) for c in nx_components}

    if manual_set == nx_set:
        print("Manual BFS verifies what NetworkX gave us")

    else:
        print("Component mismatch")

    print("family sizes:", [len(c) for c in manual_components[:]])


if __name__ == "__main__":
    main()