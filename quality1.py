import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CONFIG
from cve_dataset import CVEDataset
from cvss_metrics import CVSS_METRICS
from model import SecureBERTWithTFIDF
from train import print_metrics


def compute_all_confusion_matrices(model, test_loader, device, cvss_metrics):
    """
    Вычисляет confusion matrices для всех CVSS метрик
    
    Args:
        model: загруженная модель
        test_loader: DataLoader с тестовыми данными
        device: устройство (cuda/cpu)
        cvss_metrics: словарь с конфигурацией метрик
    
    Returns:
        dict: словарь с confusion matrices и дополнительными метриками
    """
    model.eval()
    
    # Инициализируем словари для хранения предсказаний и меток
    all_predictions = {metric: [] for metric in cvss_metrics.keys()}
    all_labels = {metric: [] for metric in cvss_metrics.keys()}
    all_confidences = {metric: [] for metric in cvss_metrics.keys()}
    
    with torch.no_grad():
        for batch in test_loader:
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            
            # Получаем предсказания модели
            outputs = model(input_ids, attention_mask, tfidf_features)
            
            # Для каждой метрики
            for metric_name in cvss_metrics.keys():
                logits = outputs[metric_name]
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Сохраняем результаты
                all_predictions[metric_name].extend(preds.cpu().numpy())
                all_labels[metric_name].extend(batch['labels'][metric_name].numpy())
                all_confidences[metric_name].extend(confidences.cpu().numpy())
    
    # Конвертируем в numpy массивы
    for metric_name in cvss_metrics.keys():
        all_predictions[metric_name] = np.array(all_predictions[metric_name])
        all_labels[metric_name] = np.array(all_labels[metric_name])
        all_confidences[metric_name] = np.array(all_confidences[metric_name])
    
    # Вычисляем confusion matrices
    confusion_matrices = {}
    classification_reports = {}
    
    for metric_name, config in cvss_metrics.items():
        cm = confusion_matrix(
            all_labels[metric_name], 
            all_predictions[metric_name],
            labels=range(len(config['classes']))
        )
        confusion_matrices[metric_name] = {
            'matrix': cm,
            'classes': config['classes'],
            'predictions': all_predictions[metric_name],
            'labels': all_labels[metric_name],
            'confidences': all_confidences[metric_name]
        }
        
        # Classification report
        report = classification_report(
            all_labels[metric_name],
            all_predictions[metric_name],
            target_names=config['classes'],
            output_dict=True,
            zero_division=0
        )
        classification_reports[metric_name] = report
    
    return confusion_matrices, classification_reports


def plot_confusion_matrix(confusion_matrix_data, metric_name, figsize=(10, 8), 
                          normalize=True, save_path=None):
    """
    Визуализация confusion matrix для одной метрики
    
    Args:
        confusion_matrix_data: словарь с матрицей и классами
        metric_name: название метрики
        figsize: размер figure
        normalize: нормализовать ли матрицу (по строкам)
        save_path: путь для сохранения (опционально)
    """
    cm = confusion_matrix_data['matrix']
    classes = confusion_matrix_data['classes']
    
    if normalize:
        # Нормализация по строкам (true labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {metric_name}'
        cbar_label = 'Proportion'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {metric_name}'
        cbar_label = 'Count'
    
    plt.figure(figsize=figsize)
    
    # Создаем heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt,
        xticklabels=classes,
        yticklabels=classes,
        cmap='Blues',
        cbar_kws={'label': cbar_label}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_all_confusion_matrices(confusion_matrices, save_dir=None, normalize=True):
    """
    Визуализация всех confusion matrices в одной сетке
    """
    n_metrics = len(confusion_matrices)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_name, cm_data) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        
        cm = cm_data['matrix']
        classes = cm_data['classes']
        
        # Нормализация
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Heatmap
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        
        # Добавляем значения
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if cm_norm[i, j] > 0.5 else "black",
                             fontsize=10)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        
        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Скрываем лишние оси
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices for All CVSS Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def print_detailed_metrics(confusion_matrices, classification_reports):
    """
    Вывод детальной статистики по каждой метрике
    """
    print("=" * 80)
    print("DETAILED METRICS FOR ALL CVSS COMPONENTS")
    print("=" * 80)
    
    for metric_name, cm_data in confusion_matrices.items():
        print(f"\n{'='*60}")
        print(f"METRIC: {metric_name.upper()}")
        print(f"{'='*60}")
        
        cm = cm_data['matrix']
        classes = cm_data['classes']
        
        # Основные метрики из classification_report
        report = classification_reports[metric_name]
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
        
        print(f"\nPer-class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
        print("-" * 70)
        
        for class_name in classes:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<20} {metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                      f"{metrics['support']:<8.0f}")
        
        # Confusion matrix в текстовом виде
        print(f"\nConfusion Matrix:")
        print(f"{'':<15}", end="")
        for class_name in classes:
            print(f"{class_name[:12]:<12}", end="")
        print()
        
        for i, class_name in enumerate(classes):
            print(f"{class_name:<15}", end="")
            for j in range(len(classes)):
                print(f"{cm[i, j]:<12}", end="")
            print()
        
        # Анализ ошибок
        print(f"\nError Analysis:")
        total_errors = cm.sum() - np.trace(cm)
        print(f"  Total errors: {total_errors} ({total_errors/cm.sum()*100:.2f}%)")
        
        # Находим самые частые ошибки
        errors = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 0:
                    errors.append({
                        'true': classes[i],
                        'pred': classes[j],
                        'count': cm[i, j]
                    })
        
        errors.sort(key=lambda x: x['count'], reverse=True)
        if errors:
            print(f"  Top 3 most common confusions:")
            for err in errors[:3]:
                print(f"    {err['true']} → {err['pred']}: {err['count']} errors")


def analyze_confidences(confusion_matrices, threshold=0.5):
    """
    Анализ уверенности модели для правильных и неправильных предсказаний
    """
    print("\n" + "=" * 80)
    print("CONFIDENCE ANALYSIS")
    print("=" * 80)
    
    for metric_name, cm_data in confusion_matrices.items():
        print(f"\n{metric_name.upper()}:")
        
        labels = cm_data['labels']
        predictions = cm_data['predictions']
        confidences = cm_data['confidences']
        
        correct_mask = (labels == predictions)
        wrong_mask = ~correct_mask
        
        if np.any(correct_mask):
            avg_conf_correct = np.mean(confidences[correct_mask])
            print(f"  Avg confidence (correct): {avg_conf_correct:.4f}")
        
        if np.any(wrong_mask):
            avg_conf_wrong = np.mean(confidences[wrong_mask])
            print(f"  Avg confidence (wrong): {avg_conf_wrong:.4f}")
        
        # Процент предсказаний выше порога
        high_conf_mask = confidences > threshold
        if np.any(high_conf_mask):
            accuracy_high_conf = np.mean(labels[high_conf_mask] == predictions[high_conf_mask])
            print(f"  Accuracy (confidence > {threshold}): {accuracy_high_conf:.4f} "
                  f"({np.sum(high_conf_mask)} samples)")


def export_results_to_csv(confusion_matrices, classification_reports, output_file='cvss_metrics_results.csv'):
    """
    Экспорт результатов в CSV файл
    """
    results_data = []
    
    for metric_name, cm_data in confusion_matrices.items():
        report = classification_reports[metric_name]
        cm = cm_data['matrix']
        
        # Добавляем метрики для каждого класса
        for class_idx, class_name in enumerate(cm_data['classes']):
            if class_name in report:
                class_metrics = report[class_name]
                results_data.append({
                    'Metric': metric_name,
                    'Class': class_name,
                    'Precision': class_metrics['precision'],
                    'Recall': class_metrics['recall'],
                    'F1-Score': class_metrics['f1-score'],
                    'Support': class_metrics['support'],
                    'Accuracy': report['accuracy'],
                    'Macro_F1': report['macro avg']['f1-score'],
                    'Weighted_F1': report['weighted avg']['f1-score']
                })
        
        # Добавляем общие метрики
        results_data.append({
            'Metric': metric_name,
            'Class': 'OVERALL',
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score'],
            'Support': cm.sum(),
            'Accuracy': report['accuracy'],
            'Macro_F1': report['macro avg']['f1-score'],
            'Weighted_F1': report['weighted avg']['f1-score']
        })
    
    df = pd.DataFrame(results_data)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results exported to {output_file}")


# ==================== ОСНОВНОЙ КОД ДЛЯ ИСПОЛЬЗОВАНИЯ ====================

def main():
    # Предполагаем, что у вас уже есть:
    # - model: загруженная модель
    # - test_loader: DataLoader с тестовыми данными
    # - device: устройство (torch.device('cuda') или torch.device('cpu'))
    # - CVSS_METRICS: словарь с конфигурацией метрик
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка модели (пример)
    # checkpoint = torch.load('cvss_model.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)
    
    # Вычисляем confusion matrices
    print("Computing confusion matrices...")
    confusion_matrices, classification_reports = compute_all_confusion_matrices(
        model, test_loader, device, CVSS_METRICS
    )
    
    # 1. Выводим детальные метрики в консоль
    print_detailed_metrics(confusion_matrices, classification_reports)
    
    # 2. Анализируем уверенность модели
    analyze_confidences(confusion_matrices, threshold=0.7)
    
    # 3. Визуализируем все confusion matrices в одной сетке
    plot_all_confusion_matrices(confusion_matrices, save_dir='./results', normalize=True)
    
    # 4. Сохраняем результаты в CSV
    export_results_to_csv(confusion_matrices, classification_reports, 'cvss_metrics_results.csv')
    
    # 5. Отдельные графики для каждой метрики (опционально)
    for metric_name, cm_data in confusion_matrices.items():
        plot_confusion_matrix(
            cm_data, 
            metric_name, 
            normalize=True,
            save_path=f'./results/confusion_matrix_{metric_name}.png'
        )
    
    return confusion_matrices, classification_reports


# Если нужно быстро получить только confusion matrix для одной метрики
def get_single_confusion_matrix(model, test_loader, device, metric_name='attack_vector'):
    """
    Быстрое получение confusion matrix для одной метрики
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            
            outputs = model(input_ids, attention_mask, tfidf_features)
            logits = outputs[metric_name]
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'][metric_name].numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_labels, all_preds


if __name__ == "__main__":
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

    def create_cvss_key(row):
        return "_".join([
            str(row['attack_vector']),
            str(row['user_interaction']),
        ])

    data['cvss_key'] = data.apply(create_cvss_key, axis=1)

    _, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['cvss_key'])

    # Инициализация токенизатора
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])

    test_dataset = CVEDataset(test_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)

    batch_size = 16  # CONFIG['BATCH_SIZE']
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loader = tqdm(test_loader, desc="Evaluating")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    model = SecureBERTWithTFIDF().to(DEVICE)

    path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=5)
    checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

    print(f"Model loaded from {path}\nValidation Metrics:")
    print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])
    model.load_state_dict(checkpoint['model_state_dict'])


    model.eval()
    
    confusion_matrices, reports = main()
