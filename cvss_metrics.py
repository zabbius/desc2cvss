
CVSS_METRICS = {
    'attack_vector': {
        'classes': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
        'classes_beta': [1.5, 3.5, 2.0, 4.0],  # beta для f-score по отдельным классам
        #'alphas': [0.009, 0.261, 0.032, 0.698],
        'alphas': [0.1, 1.5, 0.5, 2.5],
        'gamma': 1.0,
        'weight': 1.0  # вес для усреднения общего скора
    },
    'attack_complexity': {
        'classes': ['LOW', 'HIGH'],
        'classes_beta': [1.2, 3.0],
        #'alphas': [0.088, 0.912],
        'alphas': [0.2, 1.0],
        'gamma': 1.0,
        'weight': 1.0
    },
    'privileges_required': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 2.0, 3.5],
        #'alphas': [0.11, 0.198, 0.692],
        'alphas': [0.3, 0.7, 1.2],
        'gamma': 1.0,
        'weight': 1.0
    },
    'user_interaction': {
        'classes': ['NONE', 'REQUIRED'],
        'classes_beta': [1.5, 2.0],
        #'alphas': [0.344, 0.656],
        'alphas': [0.2, 1.0],
        'gamma': 1.0,
        'weight': 1.0
    },
    'scope': {
        'classes': ['UNCHANGED', 'CHANGED'],
        'classes_beta': [1.5, 2.0],
        #'alphas': [0.211, 0.789],
        'alphas': [0.15, 1.2],
        'gamma': 1.0,
        'weight': 1.0
    },
    'confidentiality': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        #'alphas': [0.448, 0.359, 0.193],
        'alphas': [0.3, 0.7, 1.2],
        'gamma': 1.0,
        'weight': 1.0
    },
    'integrity': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [2.0, 2.0, 1.2],
        #'alphas': [0.376, 0.375, 0.249],
        'alphas': [0.3, 0.7, 1.2],
        'gamma': 1.0,
        'weight': 1.0
    },
    'availability': {
        'classes': ['NONE', 'LOW', 'HIGH'],
        'classes_beta': [1.5, 3.0, 1.2],
        #'alphas': [0.19, 0.653, 0.157],
        'alphas': [0.3, 0.7, 1.2],
        'gamma': 1.0,
        'weight': 1.0
    }
}