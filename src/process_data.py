# main pipeline to process data, extract features, run graph analysis, and save outputs

import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MetaFAMLoader
from src.feature_extractor import RawFeatureExtractor
from src.graph_analysis import GraphAnalyzer
from src.inference import infer_gender, classify_anomaly_severity, is_leaf_node, is_founder_node


def print_stats(loader, features, analyzer):

    # Claude helped me a lot to format this neatly :D 
    # (it wrote this entire file but, it's just boring code here so i figured why not)

    print("\n" + "="*60)
    print("METAFAM DATASET SUMMARY")
    print("="*60)
    
    print(f"\nPeople: {len(loader.people)}")
    print(f"Triplets: {len(loader.triplets)}")
    print(f"Relation types: {len(loader.relation_types)}")
    
    # gender breakdown
    gender_counts = {'F': 0, 'M': 0, 'Unknown': 0, 'Ambiguous': 0}
    for person_id, f in features['people'].items():
        g = infer_gender(f['gender_evidence'])
        gender_counts[g['gender']] += 1
    
    print(f"\nGender distribution:")
    for g, c in gender_counts.items():
        print(f"  {g}: {c}")
    
    # generation
    gen_stats = analyzer.compute_generation_stats()
    print(f"\nGenerations: {gen_stats['min']} to {gen_stats['max']} (depth {gen_stats['depth']})")
    print(f"  distribution: {gen_stats['distribution']}")
    
    # anomalies
    anomaly_counts = {'none': 0, 'minor': 0, 'critical': 0}
    for person_id, f in features['people'].items():
        classified = classify_anomaly_severity(f['anomalies'])
        if not classified['has_anomalies']:
            anomaly_counts['none'] += 1
        elif classified['critical']:
            anomaly_counts['critical'] += 1
        else:
            anomaly_counts['minor'] += 1
    
    print(f"\nAnomaly breakdown:")
    print(f"  clean: {anomaly_counts['none']}")
    print(f"  minor issues: {anomaly_counts['minor']}")
    print(f"  critical: {anomaly_counts['critical']}")
    
    # family structure
    families = analyzer.detect_nuclear_families()
    print(f"\nNuclear families: {len(families)}")
    if families:
        sizes = [f['size'] for f in families]
        print(f"  avg size: {sum(sizes)/len(sizes):.2f}")
    
    # symmetry check
    sym_violations = analyzer.check_symmetry_violations()
    print(f"\nSymmetry violations: {len(sym_violations)}")
    
    # derivable relations
    derivable = analyzer.check_derivable_relations()
    print(f"\nDerivable relation coverage:")
    for rel, stats in derivable.items():
        if stats['total'] > 0:
            pct = stats['derivable'] / stats['total'] * 100
            print(f"  {rel}: {stats['derivable']}/{stats['total']} ({pct:.1f}%)")
    
    # high degree nodes
    high_deg = analyzer.find_high_degree_nodes(threshold=40)
    print(f"\nHigh degree nodes (40+): {len(high_deg)}")
    for node in high_deg[:5]:
        print(f"  {node['person']}: degree {node['degree']}, gen {node['generation']}")
    
    # isolated
    isolated = analyzer.find_isolated_nodes()
    print(f"\nIsolated nodes (degree <= 2): {len(isolated)}")


def save_outputs(loader, features, analyzer, output_dir='outputs'):
    """saves everything to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. raw features as json (keeps structure)
    with open(os.path.join(output_dir, 'features_raw.json'), 'w') as f:
        # need to convert sets to lists for json
        def convert(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
        json.dump(features, f, indent=2, default=convert)
    print(f"saved features_raw.json")
    
    # 2. flattened csv for easy viewing
    rows = []
    for person_id, f in features['people'].items():
        gender_inf = infer_gender(f['gender_evidence'])
        anomaly_class = classify_anomaly_severity(f['anomalies'])
        
        row = {
            'person_id': person_id,
            'gender': gender_inf['gender'],
            'gender_confidence': gender_inf['confidence'],
            'gender_female_weight': f['gender_evidence']['female_weight'],
            'gender_male_weight': f['gender_evidence']['male_weight'],
            'generation': f['generation'],
            'num_mothers': len(f['parents']['mothers']),
            'num_fathers': len(f['parents']['fathers']),
            'num_children': f['children']['total'],
            'num_siblings': len(f['siblings']),
            'degree': f['relation_counts']['total'],
            'out_degree': f['relation_counts']['total_out'],
            'in_degree': f['relation_counts']['total_in'],
            'has_anomalies': anomaly_class['has_anomalies'],
            'anomaly_count': anomaly_class.get('count', 0),
            'anomaly_types': '|'.join(anomaly_class.get('types', [])),
            'is_leaf': is_leaf_node(f),
            'is_founder': is_founder_node(f),
        }
        rows.append(row)
    
    with open(os.path.join(output_dir, 'node_features.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved node_features.csv")
    
    # 3. edges csv
    with open(os.path.join(output_dir, 'edges.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['head', 'relation', 'tail'])
        for h, r, t in loader.triplets:
            writer.writerow([h, r, t])
    print(f"saved edges.csv")
    
    # 4. anomalies detail
    anomaly_rows = []
    for person_id, f in features['people'].items():
        for a in f['anomalies']:
            anomaly_rows.append({
                'person_id': person_id,
                'type': a['type'],
                'severity': a.get('severity', 0),
                'details': json.dumps({k: v for k, v in a.items() if k not in ['type', 'severity']}),
            })
    
    if anomaly_rows:
        with open(os.path.join(output_dir, 'anomalies.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=anomaly_rows[0].keys())
            writer.writeheader()
            writer.writerows(anomaly_rows)
        print(f"saved anomalies.csv ({len(anomaly_rows)} records)")
    
    # 5. nuclear families
    families = analyzer.detect_nuclear_families()
    with open(os.path.join(output_dir, 'nuclear_families.json'), 'w') as f:
        json.dump(families, f, indent=2)
    print(f"saved nuclear_families.json ({len(families)} families)")
    
    # 6. sibling groups
    sibling_groups = analyzer.find_sibling_groups()
    with open(os.path.join(output_dir, 'sibling_groups.json'), 'w') as f:
        json.dump(sibling_groups, f, indent=2)
    print(f"saved sibling_groups.json ({len(sibling_groups)} groups)")


def main(data_path='data/train.txt'):
    
    print("loading data...")
    loader = MetaFAMLoader(data_path)
    loader.load()
    print(f"  {len(loader.triplets)} triplets, {len(loader.people)} people")
    
    print("\nextracting features...")
    extractor = RawFeatureExtractor(loader.triplets, loader.people)
    features = extractor.extract_all()
    
    print("\nrunning graph analysis...")
    analyzer = GraphAnalyzer(loader.triplets, features)
    
    print_stats(loader, features, analyzer)
    
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60 + "\n")
    save_outputs(loader, features, analyzer)
    
    print("\ndone")
    
    return loader, features, analyzer


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else '../data/train.txt'
    main(data_path)
