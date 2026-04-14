import joblib
import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CONFIG
from cve_dataset import CVEDataset
from cvss_metrics import CVSS_METRICS
from model import SecureBERTWithTFIDF
from train import print_metrics

tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

def create_cvss_key(row):
    return "_".join([
        str(row['attack_vector']),
        str(row['attack_complexity']),
        str(row['user_interaction']),
        str(row['availability']),
        str(row['scope']),
    ])

data['cvss_key'] = data.apply(create_cvss_key, axis=1)

_, test_data = train_test_split(data, test_size=0.75, random_state=42, stratify=data['cvss_key'])
_, test_data = train_test_split(test_data, test_size=0.1, random_state=42, stratify=test_data['cvss_key'])

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])

test_dataset = CVEDataset(test_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)

batch_size = 16 #CONFIG['BATCH_SIZE']
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

model = SecureBERTWithTFIDF().to(DEVICE)

path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=1)
checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

print(f"Model loaded from {path}\nValidation Metrics:")
print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])
model.load_state_dict(checkpoint['model_state_dict'])

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Предполагается, что у вас уже есть:
# - model: загруженная модель
# - test_loader: DataLoader с тестовыми данными
# - device: устройство (cuda/cpu)
# - CVSS_METRICS: словарь с конфигурацией

def get_attack_vector_predictions(model, test_loader, device):
    """
    Получение предсказаний и истинных меток для attack_vector

    Returns:
        tuple: (predictions, true_labels, probabilities)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Получаем данные из батча
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)

            # Предсказание модели
            outputs = model(input_ids, attention_mask, tfidf_features)

            # Получаем логиты для attack_vector
            logits = outputs['attack_vector']

            # Вероятности через softmax
            probabilities = torch.softmax(logits, dim=-1)

            # Предсказанные классы
            predictions = torch.argmax(probabilities, dim=-1)

            # Истинные метки
            labels = batch['labels']['attack_vector']

            # Сохраняем результаты
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def compute_confusion_matrix_attack_vector(predictions, true_labels, normalize=False):
    """
    Вычисление confusion matrix для attack_vector

    Args:
        predictions: предсказанные метки
        true_labels: истинные метки
        normalize: нормализовать ли матрицу ('true', 'pred', 'all' или False)
    """
    # Получаем количество классов
    n_classes = len(CVSS_METRICS['attack_vector']['classes'])

    # Вычисляем confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(n_classes))

    if normalize:
        if normalize == 'true':
            # Нормализация по строкам (истинным классам)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Заменяем NaN на 0
        elif normalize == 'pred':
            # Нормализация по столбцам (предсказанным классам)
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            cm = np.nan_to_num(cm)
        elif normalize == 'all':
            # Нормализация по всем элементам
            cm = cm.astype('float') / cm.sum()

    return cm


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix - Attack Vector",
                          normalize=False, save_path=None, figsize=(10, 8)):
    """
    Визуализация confusion matrix
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Выбираем цветовую схему
    if normalize:
        # Для нормализованной матрицы используем тепловую карту
        sns.heatmap(cm, annot=True, fmt='.2%' if normalize else 'd',
                    cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names, ax=ax,
                    cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    else:
        # Для абсолютных значений
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Поворачиваем метки для лучшей читаемости
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    return fig


def print_classification_report(true_labels, predictions, class_names):
    """
    Вывод detailed classification report
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT - ATTACK VECTOR")
    print("=" * 60)

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # Дополнительная статистика
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, zero_division=0
    )

    print("\n" + "-" * 60)
    print("PER-CLASS DETAILS:")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Support: {support[i]}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-Score: {f1[i]:.4f}")

    # Macro и weighted average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print("\n" + "-" * 60)
    print("AVERAGES:")
    print("-" * 60)
    print(f"Macro Average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")

    # Weighted average
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    print(
        f"Weighted Average - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")


def analyze_confusion_patterns(cm, class_names):
    """
    Анализ паттернов ошибок в confusion matrix
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS - CONFUSION PATTERNS")
    print("=" * 60)

    # Находим самые частые ошибки
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j]
                })

    errors.sort(key=lambda x: x['count'], reverse=True)

    print("\nTop 5 most frequent confusions:")
    for i, error in enumerate(errors[:5], 1):
        print(f"  {i}. {error['true']} → {error['pred']}: {error['count']} errors")

    # Анализ точности по классам
    print("\nClass-wise accuracy:")
    for i, class_name in enumerate(class_names):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = correct / total if total > 0 else 0
        print(f"  {class_name}: {accuracy:.2%} ({correct}/{total})")

    # Общая статистика ошибок
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    total_errors = total_samples - total_correct

    print(f"\nOverall statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct predictions: {total_correct} ({total_correct / total_samples:.2%})")
    print(f"  Wrong predictions: {total_errors} ({total_errors / total_samples:.2%})")


def plot_error_distribution(predictions, true_labels, class_names):
    """
    Визуализация распределения ошибок
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Распределение предсказаний vs истинных меток
    unique, true_counts = np.unique(true_labels, return_counts=True)
    unique, pred_counts = np.unique(predictions, return_counts=True)

    x = np.arange(len(class_names))
    width = 0.35

    axes[0].bar(x - width / 2, true_counts, width, label='True Labels', alpha=0.8, color='blue')
    axes[0].bar(x + width / 2, pred_counts, width, label='Predictions', alpha=0.8, color='orange')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution: True Labels vs Predictions')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy по классам
    class_accuracy = []
    for i in range(len(class_names)):
        mask = true_labels == i
        if np.any(mask):
            acc = np.mean(predictions[mask] == true_labels[mask])
        else:
            acc = 0
        class_accuracy.append(acc)

    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red'
              for acc in class_accuracy]

    bars = axes[1].bar(class_names, class_accuracy, color=colors, alpha=0.8)
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Class Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar, acc in zip(bars, class_accuracy):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{acc:.2%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def quick_confusion_matrix(model, test_loader, device):
    """
    Быстрая версия для получения confusion matrix
    """
    # Получаем class names
    class_names = CVSS_METRICS['attack_vector']['classes']

    # Получаем предсказания
    predictions, true_labels, _ = get_attack_vector_predictions(model, test_loader, device)

    # Вычисляем confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))

    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Attack Vector')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Выводим accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    return cm


def main():
    # Предполагается, что у вас уже есть:
    # model - загруженная модель
    # test_loader - DataLoader с тестовыми данными
    # device - устройство (torch.device)

    # 1. Получаем предсказания
    print("Getting predictions for attack_vector...")
    predictions, true_labels, probabilities = get_attack_vector_predictions(
        model, test_loader, DEVICE
    )

    # 2. Получаем class names из CVSS_METRICS
    class_names = CVSS_METRICS['attack_vector']['classes']

    # 3. Вычисляем confusion matrix в разных форматах
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)

    # Абсолютные значения
    cm_absolute = compute_confusion_matrix_attack_vector(predictions, true_labels, normalize=False)
    print("\nConfusion Matrix (absolute values):")
    print(cm_absolute)

    # Нормализованная по строкам (процент ошибок в каждом истинном классе)
    cm_normalized_true = compute_confusion_matrix_attack_vector(predictions, true_labels, normalize='true')
    print("\nConfusion Matrix (normalized by true class):")
    print(np.round(cm_normalized_true, 3))

    # Нормализованная по столбцам (процент предсказаний в каждом предсказанном классе)
    cm_normalized_pred = compute_confusion_matrix_attack_vector(predictions, true_labels, normalize='pred')
    print("\nConfusion Matrix (normalized by predicted class):")
    print(np.round(cm_normalized_pred, 3))

    # 4. Визуализация
    # Абсолютная матрица
    plot_confusion_matrix(
        cm_absolute,
        class_names,
        title="Confusion Matrix - Attack Vector (Absolute Counts)",
        normalize=False,
        save_path="confusion_matrix_absolute.png"
    )

    # Нормализованная матрица
    plot_confusion_matrix(
        cm_normalized_true,
        class_names,
        title="Confusion Matrix - Attack Vector (Normalized by True Class)",
        normalize=True,
        save_path="confusion_matrix_normalized.png"
    )

    # 5. Classification report
    print_classification_report(true_labels, predictions, class_names)

    # 6. Анализ ошибок
    analyze_confusion_patterns(cm_absolute, class_names)

    # 7. Визуализация распределения ошибок
    plot_error_distribution(predictions, true_labels, class_names)

    # 8. Дополнительный анализ: уверенность модели для правильных и неправильных предсказаний
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    # Получаем максимальные вероятности для каждого предсказания
    max_probabilities = np.max(probabilities, axis=1)

    # Разделяем правильные и неправильные предсказания
    correct_mask = predictions == true_labels
    wrong_mask = ~correct_mask

    print(f"\nAverage confidence:")
    print(f"  Correct predictions: {np.mean(max_probabilities[correct_mask]):.4f}")
    print(f"  Wrong predictions: {np.mean(max_probabilities[wrong_mask]):.4f}")

    # Анализ низкой уверенности
    low_confidence_threshold = 0.6
    low_conf_mask = max_probabilities < low_confidence_threshold

    print(f"\nLow confidence predictions (<{low_confidence_threshold}):")
    print(f"  Count: {np.sum(low_conf_mask)} ({np.mean(low_conf_mask):.2%})")
    if np.any(low_conf_mask):
        low_conf_accuracy = np.mean(predictions[low_conf_mask] == true_labels[low_conf_mask])
        print(f"  Accuracy on low confidence: {low_conf_accuracy:.4f}")

    # 9. Сохраняем результаты в файл
    results = {
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'confusion_matrix_absolute': cm_absolute.tolist(),
        'confusion_matrix_normalized': cm_normalized_true.tolist(),
        'classification_report': classification_report(true_labels, predictions,
                                                       target_names=class_names,
                                                       output_dict=True)
    }

    import json
    with open('attack_vector_results.json', 'w') as f:
        # Конвертируем numpy типы в Python типы
        json.dump(results, f, indent=2)
    print("\nResults saved to 'attack_vector_results.json'")

    return predictions, true_labels, probabilities, cm_absolute, cm_normalized_true

#predictions, true_labels, probabilities, cm_abs, cm_norm = main()
cm = quick_confusion_matrix(model, test_loader, DEVICE)
