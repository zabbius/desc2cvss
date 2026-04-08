import numpy as np
import torch
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm

from cvss_metrics import CVSS_METRICS
from model import ApproxFBetaLoss


def compute_metrics(predictions, labels, metric_name):
    """Вычисление accuracy, FBeta, recall, precision для конкретной метрики с учетом beta и весов классов"""
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    true = labels.cpu().numpy()

    config = CVSS_METRICS[metric_name]
    beta = config['classes_beta']
    num_classes = len(config['classes'])

    # Вычисляем метрики для каждого класса отдельно
    fbeta_scores = []
    recall_scores = []
    precision_scores = []
    class_accuracies = []

    for class_idx in range(num_classes):
        # Создаем бинарные метки для текущего класса
        true_binary = (true == class_idx).astype(int)
        preds_binary = (preds == class_idx).astype(int)

        # Проверяем, есть ли положительные примеры
        if np.sum(true_binary) == 0 and np.sum(preds_binary) == 0:
            fbeta_scores.append(1.0)  # Нет примеров класса - считаем идеальным
            recall_scores.append(1.0)
            precision_scores.append(1.0)
        else:
            fbeta = fbeta_score(true_binary, preds_binary, beta=beta[class_idx], zero_division=0)
            recall = recall_score(true_binary, preds_binary, zero_division=0)
            precision = precision_score(true_binary, preds_binary, zero_division=0)

            fbeta_scores.append(fbeta)
            recall_scores.append(recall)
            precision_scores.append(precision)

        # Accuracy для класса (доля правильных предсказаний для этого класса)
        if np.sum(true_binary) == 0:
            class_accuracies.append(1.0)  # Нет примеров класса
        else:
            correct = np.sum((true_binary == 1) & (preds_binary == 1))
            total = np.sum(true_binary)
            class_accuracies.append(correct / total if total > 0 else 1.0)

    weighted_fbeta = np.average(fbeta_scores)
    weighted_recall = np.average(recall_scores)
    weighted_precision = np.average(precision_scores)
    weighted_class_accuracy = np.average(class_accuracies)

    # Общая accuracy (не взвешенная, просто доля правильных ответов)
    overall_accuracy = accuracy_score(true, preds)

    return {
        'accuracy': overall_accuracy,
        'weighted_accuracy': weighted_class_accuracy,
        'fbeta': weighted_fbeta,
        'recall': weighted_recall,
        'precision': weighted_precision,
        'per_class_fbeta': fbeta_scores,
        'per_class_recall': recall_scores,
        'per_class_precision': precision_scores
    }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0

    loss_fn = ApproxFBetaLoss().to(device)

    progress_bar = tqdm(dataloader, desc="Training", dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress_bar):
        # Перемещаем на устройство
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tfidf_features = batch['tfidf_features'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # Forward pass
        outputs = model(input_ids, attention_mask, tfidf_features)

        loss, _ = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        current_loss = loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Оценка модели"""
    model.eval()
    metrics = {}

    all_predictions = {metric: [] for metric in CVSS_METRICS.keys()}
    all_labels = {metric: [] for metric in CVSS_METRICS.keys()}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask, tfidf_features)

            for metric_name in CVSS_METRICS.keys():
                all_predictions[metric_name].append(outputs[metric_name].cpu())
                all_labels[metric_name].append(labels[metric_name])

    # Вычисляем метрики для каждой задачи
    for metric_name in CVSS_METRICS.keys():
        preds = torch.cat(all_predictions[metric_name], dim=0)
        labels = torch.cat(all_labels[metric_name], dim=0)

        task_metrics = compute_metrics(preds, labels, metric_name)
        metrics[metric_name] = task_metrics

    return metrics

def reduce_metrics(metrics):
    weighted_avg_fbeta = 0
    weighted_avg_recall = 0
    weighted_avg_precision = 0
    weighted_avg_accuracy = 0
    total_weight = 0

    for metric_name, metric_values in metrics.items():
        weight = CVSS_METRICS[metric_name]['weight']
        weighted_fbeta = metric_values['fbeta'] * weight
        weighted_recall = metric_values['recall'] * weight
        weighted_precision = metric_values['precision'] * weight
        weighted_accuracy = metric_values['weighted_accuracy'] * weight

        weighted_avg_fbeta += weighted_fbeta
        weighted_avg_recall += weighted_recall
        weighted_avg_precision += weighted_precision
        weighted_avg_accuracy += weighted_accuracy
        total_weight += weight

    weighted_avg_fbeta /= total_weight
    weighted_avg_recall /= total_weight
    weighted_avg_precision /= total_weight
    weighted_avg_accuracy /= total_weight

    return {
        'weighted_avg_fbeta': weighted_avg_fbeta,
        'weighted_avg_recall': weighted_avg_recall,
        'weighted_avg_precision': weighted_avg_precision,
        'weighted_avg_accuracy': weighted_avg_accuracy,
    }


def print_metrics(metrics, reduced_metrics):
    for metric_name, metric_values in metrics.items():
        weight = CVSS_METRICS[metric_name]['weight']
        print(f"  {metric_name}:")
        print(
            f"    Acc={metric_values['accuracy']:.4f} (weighted={metric_values['weighted_accuracy']:.4f}), FBeta={metric_values['fbeta']:.4f}, Recall={metric_values['recall']:.4f}, Precision={metric_values['precision']:.4f} (weight={weight})")

    print(f"\n  Weighted Averages:")
    print(f"    Accuracy: {reduced_metrics['weighted_avg_accuracy']:.4f}")
    print(f"    FBeta: {reduced_metrics['weighted_avg_fbeta']:.4f}")
    print(f"    Recall: {reduced_metrics['weighted_avg_recall']:.4f}")
    print(f"    Precision: {reduced_metrics['weighted_avg_precision']:.4f}")
