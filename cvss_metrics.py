
CVSS_METRICS = {
    'attack_vector': {
        'classes': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
        'classes_beta': [1.5, 3.5, 2.0, 4.0],  # beta для f-score по отдельным классам
        'alphas': [0.1, 2.0, 0.4, 3.0],
        #'alphas': [0.1, 1.5, 0.5, 2.5],
        'gamma': 2.0,
        'weight': 1.0  # вес для усреднения общего скора
    },
    'attack_complexity': {
        'classes': ['LOW', 'HIGH'],
        'classes_beta': [1.2, 3.0],
        'alphas': [0.15, 1.5],
        #'alphas': [0.2, 1.0],
        'gamma': 2.0,
        'weight': 1.0
    },
    'privileges_required': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 2.0, 3.5],
        'alphas': [0.2, 0.8, 1.5],
        #'alphas': [0.3, 0.7, 1.2],
        'gamma': 2.0,
        'weight': 1.0
    },
    'user_interaction': {
        'classes': ['NONE', 'REQUIRED'],
        'classes_beta': [1.5, 2.0],
        'alphas': [0.15, 1.5],
        #'alphas': [0.2, 1.0],
        'gamma': 2.0,
        'weight': 1.0
    },
    'scope': {
        'classes': ['UNCHANGED', 'CHANGED'],
        'classes_beta': [1.5, 2.0],
        'alphas':  [0.1, 1.5],
        #'alphas': [0.15, 1.2],
        'gamma': 2.0,
        'weight': 1.0
    },
    'confidentiality': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        'alphas': [0.2, 0.8, 1.5],
        #'alphas': [0.3, 0.7, 1.2],
        'gamma': 2.0,
        'weight': 1.0
    },
    'integrity': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        'alphas': [0.2, 0.8, 1.5],
        #'alphas': [0.3, 0.7, 1.2],
        'gamma': 2.0,
        'weight': 1.0
    },
    'availability': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 3.0, 1.2],
        'alphas': [0.2, 0.8, 1.5],
        #'alphas': [0.3, 0.7, 1.2],
        'gamma': 2.0,
        'weight': 1.0
    }
}