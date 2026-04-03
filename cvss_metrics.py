
CVSS_METRICS = {
    'attack_vector': {
        'classes': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
        'classes_weights': [1.0, 1.0, 1.0, 1.0],  # веса для усреднения метрики
        'classes_beta': [1.0, 1.0, 1.0, 1.0],  # beta для f-score по отдельным классам
        'weight': 1.0  # вес для усреднения общего скора
    },
    'attack_complexity': {
        'classes': ['LOW', 'HIGH'],
        'classes_weights': [1.0, 1.0],
        'classes_beta': [1.0, 1.0],
        'weight': 1.0
    },
    'privileges_required': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.0, 1.0, 1.0],
        'classes_beta': [1.0, 1.0, 1.0],
        'weight': 1.5  # Больший вес из-за дисбаланса
    },
    'user_interaction': {
        'classes': ['NONE', 'REQUIRED'],
        'classes_weights': [1.0, 1.0],
        'classes_beta': [1.0, 1.0],
        'weight': 1.0
    },
    'scope': {
        'classes': ['UNCHANGED', 'CHANGED'],
        'classes_weights': [1.0, 1.0],
        'classes_beta': [1.0, 1.0],
        'weight': 1.0
    },
    'confidentiality': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.0, 1.0, 1.0],
        'classes_beta': [1.0, 1.0, 1.0],
        'num_classes': 3,
        'weight': 1.2
    },
    'integrity': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.0, 1.0, 1.0],
        'classes_beta': [1.0, 1.0, 1.0],
        'weight': 1.2
    },
    'availability': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_weights': [1.0, 1.0, 1.0],
        'classes_beta': [1.0, 1.0, 1.0],
        'weight': 1.2
    }
}
