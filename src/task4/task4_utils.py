# shared utilities for task 4 link prediction

import numpy as np # type: ignore
import torch # pyright: ignore[reportMissingImports]
from collections import defaultdict




def load_triplets(filepath):
    
    triplets = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append(tuple(parts))
    return triplets


def build_mappings(train_triplets, test_triplets):

    # assign integer IDs to every entity and relation we see
    
    entities = set()
    relations = set()
    for h, r, t in train_triplets + test_triplets:
        entities.update([h, t])
        relations.add(r)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    id2entity = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}

    return entity2id, relation2id, id2entity, id2relation


# CLAUDE ASSISTED CODE BEGINS !!!!!!!!!!!!!!!!!!

def triplets_to_ids(triplets, entity2id, relation2id):

    ids = []
    for h, r, t in triplets:
        ids.append([entity2id[h], relation2id[r], entity2id[t]])
    return np.array(ids, dtype=np.int64)


# CLAUDE ASSISTED CODE ENDS !!!!!!!!!!!!!!!!!!

def load_all_data(train_path, test_path):
    
    # returns a dict with all da pieces

    train_raw = load_triplets(train_path)
    test_raw = load_triplets(test_path)

    entity2id, relation2id, id2entity, id2relation = build_mappings(train_raw, test_raw)

    train_ids = triplets_to_ids(train_raw, entity2id, relation2id)
    test_ids = triplets_to_ids(test_raw, entity2id, relation2id)

    # all known triplets as a set for filtered evaluation
    all_triplets_set = set()
    for row in train_ids:
        all_triplets_set.add(tuple(row))
    for row in test_ids:
        all_triplets_set.add(tuple(row))

    return {
        'train_raw': train_raw, 'test_raw': test_raw,
        'train_ids': train_ids, 'test_ids': test_ids,
        'entity2id': entity2id, 'relation2id': relation2id,
        'id2entity': id2entity, 'id2relation': id2relation,
        'num_entities': len(entity2id), 'num_relations': len(relation2id),
        'all_triplets_set': all_triplets_set,
    }


# CLAUDE ASSISTED CODE BEGINS !!!!!!!!!!!!!!!!!!

def negative_sample(pos_triplets, num_entities, neg_ratio=10):
    """
    for each positive triplet, create neg_ratio corrupted ones
    by randomly replacing head or tail.
    returns (all_triplets, labels) where labels are 1 for positive, 0 for negative.
    """
    n = len(pos_triplets)
    neg_triplets = np.tile(pos_triplets, (neg_ratio, 1))

    # randomly corrupt head or tail
    corrupt_head = np.random.random(n * neg_ratio) > 0.5
    random_entities = np.random.randint(0, num_entities, size=n * neg_ratio)
    neg_triplets[corrupt_head, 0] = random_entities[corrupt_head]
    neg_triplets[~corrupt_head, 2] = random_entities[~corrupt_head]

    all_triplets = np.concatenate([pos_triplets, neg_triplets])
    labels = np.zeros(len(all_triplets), dtype=np.float32)
    labels[:n] = 1.0

    return all_triplets, labels


def evaluate_link_prediction(entity_emb, rel_emb, test_triplets, all_true_set,
                              num_entities, score_fn='distmult', batch_size=64,
                              verbose=True):
    """
    standard filtered evaluation for link prediction.

    for each test triplet (h, r, t):
      - tail prediction: score (h, r, e) for all entities e, filter known triples, rank t
      - head prediction: score (e, r, t) for all entities e, filter known triples, rank h
    
    score_fn: 'distmult' uses sum(h * r * t)

    returns dict with MRR, Hits@1, Hits@3, Hits@10 and per-triplet ranks
    """
    entity_emb = torch.as_tensor(entity_emb, dtype=torch.float32)
    rel_emb = torch.as_tensor(rel_emb, dtype=torch.float32)

    all_ranks = []

    for start in range(0, len(test_triplets), batch_size):
        batch = test_triplets[start:start + batch_size]

        for h, r, t in batch:
            # ── tail prediction: (h, r, ?) ──
            h_emb = entity_emb[h].unsqueeze(0)       # (1, dim)
            r_emb = rel_emb[r].unsqueeze(0)           # (1, dim)
            all_t_emb = entity_emb                     # (num_entities, dim)

            if score_fn == 'distmult':
                scores = torch.sum(h_emb * r_emb * all_t_emb, dim=1)  # (num_entities,)

            # filter out known true triplets (except the one we're testing)
            for e in range(num_entities):
                if e != t and (h, r, e) in all_true_set:
                    scores[e] = -1e9

            rank_t = int((scores >= scores[t]).sum().item())
            all_ranks.append(rank_t)

            # ── head prediction: (?, r, t) ──
            t_emb = entity_emb[t].unsqueeze(0)
            all_h_emb = entity_emb

            if score_fn == 'distmult':
                scores = torch.sum(all_h_emb * r_emb * t_emb, dim=1)

            for e in range(num_entities):
                if e != h and (e, r, t) in all_true_set:
                    scores[e] = -1e9

            rank_h = int((scores >= scores[h]).sum().item())
            all_ranks.append(rank_h)

        if verbose and start % 200 == 0:
            print(f"  evaluated {len(test_triplets)} triplets")

    ranks = np.array(all_ranks, dtype=np.float64)
    metrics = compute_metrics(ranks)

    if verbose:
        print(f"\n  MRR:     {metrics['mrr']:.4f}")
        print(f"  Hits@1:  {metrics['hits@1']:.4f}")
        print(f"  Hits@3:  {metrics['hits@3']:.4f}")
        print(f"  Hits@10: {metrics['hits@10']:.4f}")

    return {**metrics, 'all_ranks': ranks}



# CLAUDE ASSISTED CODE ENDS !!!!!!!!!!!!!!!!!!


def compute_metrics(ranks):
    
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)
    return {'mrr': mrr, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'mean_rank': mean_rank}


def evaluate_per_relation(entity_emb, rel_emb, test_triplets, all_true_set,
                           num_entities, id2relation, score_fn='distmult'):


    # TODO MADE THIS COMPATIBLE WITH THE RGCN ALSO
    
    entity_emb = torch.as_tensor(entity_emb, dtype=torch.float32)
    rel_emb = torch.as_tensor(rel_emb, dtype=torch.float32)

    ranks_by_rel = defaultdict(list)

    for h, r, t in test_triplets:
        # tail prediction
        h_emb = entity_emb[h].unsqueeze(0)
        r_emb_vec = rel_emb[r].unsqueeze(0)

        if score_fn == 'distmult':
            scores = torch.sum(h_emb * r_emb_vec * entity_emb, dim=1)
        for e in range(num_entities):
            if e != t and (h, r, e) in all_true_set:
                scores[e] = -1e9
        rank_t = int((scores >= scores[t]).sum().item())

        # head prediction
        t_emb = entity_emb[t].unsqueeze(0)
        if score_fn == 'distmult':
            scores = torch.sum(entity_emb * r_emb_vec * t_emb, dim=1)
        for e in range(num_entities):
            if e != h and (e, r, t) in all_true_set:
                scores[e] = -1e9
        rank_h = int((scores >= scores[h]).sum().item())

        rel_name = id2relation[r]
        ranks_by_rel[rel_name].extend([rank_t, rank_h])

    results = {}
    for rel_name, ranks in ranks_by_rel.items():
        results[rel_name] = compute_metrics(np.array(ranks, dtype=np.float64))

    return results
