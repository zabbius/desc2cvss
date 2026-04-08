
CVSS_METRICS = {
    'attack_vector': {
        'classes': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
        'classes_beta': [0.5, 0.6, 0.5, 0.9],  # beta для f-score по отдельным классам
        'weight': 1.5  # вес для усреднения общего скора
    },
    'attack_complexity': {
        'classes': ['LOW', 'HIGH'],
        'classes_beta': [1.2, 3.0],
        'weight': 1.0
    },
    'privileges_required': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 2.0, 3.5],
        'weight': 1.0
    },
    'user_interaction': {
        'classes': ['NONE', 'REQUIRED'],
        'classes_beta': [1.5, 2.0],
        'weight': 1.0
    },
    'scope': {
        'classes': ['UNCHANGED', 'CHANGED'],
        'classes_beta': [1.5, 2.0],
        'weight': 1.0
    },
    'confidentiality': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        'weight': 1.0
    },
    'integrity': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        'weight': 1.0
    },
    'availability': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 3.0, 1.2],
        'weight': 1.0
    }
}