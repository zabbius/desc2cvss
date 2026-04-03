import torch

from sklearn.metrics import fbeta_score, accuracy_score
from torch import nn
from tqdm import tqdm

from cvss_metrics import CVSS_METRICS

def compute_metrics(predictions, labels):
    """Вычисление accuracy и F1 для конкретной метрики"""
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    true = labels.cpu().numpy()

    return {
        'accuracy': accuracy_score(true, preds),
        'fbeta': fbeta_score(true, preds, beta=1, average='macro')
    }

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0
    all_predictions = {metric: [] for metric in CVSS_METRICS.keys()}
    all_labels = {metric: [] for metric in CVSS_METRICS.keys()}

    for batch in tqdm(dataloader, desc="Training"):
        # Перемещаем на устройство
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tfidf_features = batch['tfidf_features'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # Forward pass
        outputs = model(input_ids, attention_mask, tfidf_features)

        # Вычисляем потери для каждой задачи
        task_losses = []
        for metric_name in CVSS_METRICS.keys():
            loss = nn.functional.cross_entropy(outputs[metric_name], labels[metric_name])
            task_losses.append(loss)

            # Сохраняем для метрик
            all_predictions[metric_name].append(outputs[metric_name].detach())
            all_labels[metric_name].append(labels[metric_name])

        # Multi-task loss с автоматической настройкой весов
        loss = criterion(task_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Оценка модели"""
    model.eval()
    metrics = {metric: {'accuracy': 0, 'f1': 0} for metric in CVSS_METRICS.keys()}

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

        task_metrics = compute_metrics(preds, labels)
        metrics[metric_name] = task_metrics

    return metrics


