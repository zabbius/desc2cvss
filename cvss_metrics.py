
CVSS_METRICS = {
    'attack_vector': {
        'classes': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
        'classes_weights': [1.2, 3.6, 2.1, 4.6],  # веса для усреднения метрики
        'classes_beta': [1.5, 3.5, 2.0, 4.0],  # beta для f-score по отдельным классам
        'weight': 1.0  # вес для усреднения общего скора
    },
    'attack_complexity': {
        'classes': ['LOW', 'HIGH'],
        'classes_weights': [1.7, 2.8],
        'classes_beta': [1.2, 3.0],
        'weight': 1.0
    },
    'privileges_required': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [2.0, 2.2, 3.1],
        'classes_beta': [1.5, 2.0, 3.5],
        'weight': 1.0
    },
    'user_interaction': {
        'classes': ['NONE', 'REQUIRED'],
        'classes_weights': [1.9, 2.1],
        'classes_beta': [1.5, 2.0],
        'weight': 1.0
    },
    'scope': {
        'classes': ['UNCHANGED', 'CHANGED'],
        'classes_weights': [1.8, 2.5],
        'classes_beta': [1.5, 2.0],
        'weight': 1.0
    },
    'confidentiality': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.7, 2.2, 2.1],
        'classes_beta': [2.0, 2.0, 1.2],
        'weight': 1.0
    },
    'integrity': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.5, 2.1, 2.2],
        'classes_beta': [2.0, 2.0, 1.2],
        'weight': 1.0
    },
    'availability': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.3, 2.8, 2.1],
        'classes_beta': [1.5, 3.0, 1.2],
        'weight': 1.0
    }
}