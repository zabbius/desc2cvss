import torch
import torch.nn as nn
from transformers import AutoModel

from config import CONFIG
from cvss_metrics import CVSS_METRICS


class SecureBERTWithTFIDF(nn.Module):
    """
    Multi-Task Learning модель для предсказания 8 метрик CVSS
    с fusion слоем для объединения эмбеддингов BERT и TF-IDF
    """

    def __init__(self):
        super().__init__()

        # SecureBERT encoder
        self.bert = AutoModel.from_pretrained(CONFIG['MODEL_NAME'])
        self.bert_hidden_size = self.bert.config.hidden_size

        # TF-IDF проекция (сжимаем до меньшей размерности)
        self.tfidf_projection = nn.Linear(CONFIG['TFIDF_MAX_FEATURES'], CONFIG['TFIDF_FUSION_SIZE'])
        self.tfidf_dropout = nn.Dropout(0.3)

        # Fusion слой для объединения BERT и TF-IDF
        fusion_input_size = self.bert_hidden_size + CONFIG['TFIDF_FUSION_SIZE']
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Классификационные головки для каждой метрики (Multi-Task)
        self.classifiers = nn.ModuleDict()
        for metric_name, cvss_config in CVSS_METRICS.items():
            self.classifiers[metric_name] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, len(cvss_config['classes']))
            )

    def forward(self, input_ids, attention_mask, tfidf_features):
        # Получаем эмбеддинги из SecureBERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Используем [CLS] токен (первый токен) как представление всего текста
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # Проецируем TF-IDF признаки
        tfidf_projected = self.tfidf_projection(tfidf_features)  # [batch_size, 256]
        tfidf_projected = self.tfidf_dropout(tfidf_projected)

        # Конкатенируем BERT и TF-IDF представления
        combined = torch.cat([cls_embeddings, tfidf_projected], dim=1)  # [batch_size, 1024]

        # Fusion слой
        fused_features = self.fusion_layer(combined)  # [batch_size, 256]

        # Предсказания для каждой метрики
        outputs = {}
        for metric_name, classifier in self.classifiers.items():
            outputs[metric_name] = classifier(fused_features)

        return outputs

class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss с автоматической настройкой весов задач
    на основе неопределенности (Kendall et al., 2018)
    """

    def __init__(self, num_tasks):
        super().__init__()
        # Обучаемые параметры log(sigma^2) для каждой задачи
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i] / 2
        return total_loss

class ApproxFBetaLoss(nn.Module):
    """
    Упрощенная версия для прямой оптимизации f-beta score
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        """
        Вычисление взвешенного f-beta loss

        Returns:
            total_loss: взвешенный f-beta loss
            component_losses: словарь с loss для каждой компоненты
        """
        total_loss = 0.0
        component_losses = {}

        for metric_name, logits in outputs.items():
            config = CVSS_METRICS[metric_name]
            targets_metric = targets[metric_name]

            # Получаем вероятности
            probs = torch.softmax(logits, dim=1)
            targets_one_hot = torch.zeros_like(probs).scatter_(1, targets_metric.unsqueeze(1), 1)

            # Вычисляем weighted f-beta для компоненты
            per_class_fbeta = []

            for class_idx, (beta, class_weight) in enumerate(zip(config['classes_beta'], config['classes_weights'])):
                p = probs[:, class_idx]
                t = targets_one_hot[:, class_idx]

                # Мягкие метрики
                true_positives = (p * t).sum()
                predicted_positives = p.sum()
                actual_positives = t.sum()

                precision = true_positives / (predicted_positives + self.epsilon)
                recall = true_positives / (actual_positives + self.epsilon)

                beta_squared = beta ** 2
                fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + self.epsilon)

                # Применяем вес класса
                per_class_fbeta.append(fbeta * class_weight)

            # Взвешенное среднее по классам для компоненты
            total_class_weight = sum(config['classes_weights'])
            component_fbeta = sum(per_class_fbeta) / total_class_weight

            # Loss = 1 - fbeta
            component_loss = 1 - component_fbeta

            # Применяем вес компоненты
            weighted_component_loss = component_loss * config['weight']
            total_loss += weighted_component_loss

            component_losses[metric_name] = component_loss.detach().cpu().item()

        return total_loss, component_losses
