import numpy as np
import pandas

from collections import Counter

from cvss_metrics import CVSS_METRICS

import pandas as pd
import numpy as np
from collections import Counter


def compute_alphas_from_csv(csv_path, cvss_metrics, method='inverse_frequency'):
    """
    Вычисление alpha для всех CVSS метрик из CSV файла

    Args:
        csv_path: путь к CSV файлу
        cvss_metrics: словарь CVSS_METRICS
        method: 'inverse_frequency', 'effective_number', или 'sqrt'

    Returns:
        dict: alpha веса для каждой метрики
    """
    # Загрузка данных
    df = pd.read_csv(csv_path)

    alphas = {}

    for metric_name in cvss_metrics.keys():
        # Проверяем, есть ли колонка в данных
        if metric_name not in df.columns:
            print(f"Warning: {metric_name} not found in CSV")
            continue

        # Получаем метки
        labels = df[metric_name].values

        # Подсчитываем частоту классов
        class_counts = Counter(labels)
        n_classes = len(cvss_metrics[metric_name]['classes'])
        total = len(labels)

        # Инициализируем alpha нулями
        alpha = np.zeros(n_classes)

        for class_idx in range(n_classes):
            count = class_counts.get(class_idx, 0)

            if count == 0:
                # Класс отсутствует в данных
                alpha[class_idx] = 0.0
                continue

            if method == 'inverse_frequency':
                alpha[class_idx] = total / (n_classes * count)

            elif method == 'effective_number':
                beta = 0.999
                alpha[class_idx] = (1 - beta) / (1 - beta ** count)

            elif method == 'sqrt':
                alpha[class_idx] = np.sqrt(total / count)

        # Нормализация (если сумма > 0)
        if alpha.sum() > 0:
            alpha = alpha / alpha.sum()

        alphas[metric_name] = alpha.tolist()

        # Выводим статистику
        print(f"\n{metric_name}:")
        print(f"  Distribution: {dict(class_counts)}")
        print(f"  Alpha weights: {[round(x, 3) for x in alpha.tolist()]}")

    return alphas


alpha_dict = compute_alphas_from_csv(
    "data/filtered_output_ML_all.csv.gz",
    CVSS_METRICS,
    method='inverse_frequency'
)

print(alpha_dict)