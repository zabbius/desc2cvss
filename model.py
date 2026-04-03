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
