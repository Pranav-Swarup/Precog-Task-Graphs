# inference functions that make decisions from raw features
# all the thresholding, rounding, classification happens here
# keeping this separate so raw data stays clean

def infer_gender(gender_evidence: dict, threshold: float = 0.7) -> dict:


    # here the data was actually clean. and all people turned out with 1.0 confidence dude or girl but itll work with broken data too
    # makes gender decision from raw evidence
    # threshold = what ratio counts as confident. 

    f = gender_evidence['female_weight']
    m = gender_evidence['male_weight']
    total = f + m
    
    if total == 0:
        return {
            'gender': 'Unknown',
            'confidence': 0.0,
            'reason': 'no_evidence',
        }
    
    ratio = gender_evidence['female_ratio']
    
    if ratio >= threshold:
        return {
            'gender': 'F',
            'confidence': ratio,
            'reason': 'female_dominant',
        }
    elif ratio <= (1 - threshold):
        return {
            'gender': 'M', 
            'confidence': 1 - ratio,
            'reason': 'male_dominant',
        }
    else:
        # mixed evidence -> gives details
        return {
            'gender': 'Ambiguous',
            'confidence': max(ratio, 1 - ratio),
            'reason': 'mixed_evidence',
            'female_weight': f,
            'male_weight': m,
        }


def classify_anomaly_severity(anomalies: list) -> dict:
 

    if not anomalies:
        return {'has_anomalies': False, 'max_severity': 0, 'types': []}
    
    types = [a['type'] for a in anomalies]
    severities = [a.get('severity', 0) for a in anomalies]
    
    return {
        'has_anomalies': True,
        'count': len(anomalies),
        'max_severity': max(severities),
        'avg_severity': sum(severities) / len(severities),
        'types': types,
        'critical': any(s >= 1.0 for s in severities),
    }


def is_leaf_node(person_features: dict) -> bool:
    # no kids
    return person_features['children']['total'] == 0


def is_founder_node(person_features: dict) -> bool:
    #orphans or have no recorded parents
    return person_features['parents']['total'] == 0


def is_bridge_candidate(person_features: dict) -> bool:
    
    # has both parents and children
    # TODO make this a full blown heuristics analysis where I look at counts and generations and stuff 
    # need to find people who are key players who connect different famly clusters

    has_parents = person_features['parents']['total'] > 0
    has_children = person_features['children']['total'] > 0
    return has_parents and has_children
