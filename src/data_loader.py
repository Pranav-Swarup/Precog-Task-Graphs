from collections import defaultdict

class MetaFAMLoader:

    def __init__(self, filepath: str):
        
        self.filepath = filepath
        self.triplets = []
        self.people = set()
        self.relation_types = set()
        
    def load(self):

        with open(self.filepath, 'r') as f:
        
            for line in f:
                line = line.strip()
                if not line:
                    continue
        
                parts = line.split()
                if len(parts) == 3:
                    head, relation, tail = parts
                    self.triplets.append((head, relation, tail))
                    self.people.add(head)
                    self.people.add(tail)
                    self.relation_types.add(relation)
        
        return self.triplets
    
    def build_adjacency(self):

        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        
        for head, rel, tail in self.triplets:
            outgoing[head].append((rel, tail))
            incoming[tail].append((rel, head))
        
        return outgoing, incoming
