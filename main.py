import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CONFIG
from cve_dataset import CVEDataset
from cvss_metrics import CVSS_METRICS
from model import SecureBERTWithTFIDF
from train import evaluate, train_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])

# Создание TF-IDF векторайзера на тренировочных данных
print("Fitting TF-IDF vectorizer...")
#data = pandas.read_csv("data/output_ML_all.csv.gz")
data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, stratify=data['year'])
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['year'])

train_dataset = CVEDataset(train_data, tokenizer)
tfidf_vectorizer = train_dataset.tfidf_vectorizer

val_dataset = CVEDataset(val_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)
test_dataset = CVEDataset(test_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

# Инициализация модели
model = SecureBERTWithTFIDF().to(DEVICE)
# Оптимизатор и scheduler
optimizer = AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])

total_steps = len(train_loader) * CONFIG['EPOCHS']

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=CONFIG['WARMUP_STEPS'], num_training_steps=total_steps)

# В основном цикле обучения:
for epoch in range(CONFIG['EPOCHS']):
    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
    print(f"{'=' * 50}")

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    metrics = evaluate(model, val_loader, DEVICE)

    # Вывод результатов с учетом weight каждой метрики
    print("\nValidation Metrics:")
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

        print(f"  {metric_name}:")
        print(
            f"    Acc={metric_values['accuracy']:.4f} (weighted={metric_values['weighted_accuracy']:.4f}), FBeta={metric_values['fbeta']:.4f}, Recall={metric_values['recall']:.4f}, Precision={metric_values['precision']:.4f} (weight={weight})")

        weighted_avg_fbeta += weighted_fbeta
        weighted_avg_recall += weighted_recall
        weighted_avg_precision += weighted_precision
        weighted_avg_accuracy += weighted_accuracy
        total_weight += weight

    weighted_avg_fbeta /= total_weight
    weighted_avg_recall /= total_weight
    weighted_avg_precision /= total_weight
    weighted_avg_accuracy /= total_weight

    print(f"\n  Weighted Averages:")
    print(f"    Accuracy: {weighted_avg_accuracy:.4f}")
    print(f"    FBeta: {weighted_avg_fbeta:.4f}")
    print(f"    Recall: {weighted_avg_recall:.4f}")
    print(f"    Precision: {weighted_avg_precision:.4f}")

    path = f"cvss_model_epoch_{epoch + 1:02d}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tfidf_vectorizer': tfidf_vectorizer,
        'metrics': metrics,
        'weighted_avg_fbeta': weighted_avg_fbeta,
        'weighted_avg_recall': weighted_avg_recall,
        'weighted_avg_precision': weighted_avg_precision,
        'weighted_avg_accuracy': weighted_avg_accuracy,
    }, path)
    print(f"✓ Model saved to {path}")

print(f"\nTraining complete!")
