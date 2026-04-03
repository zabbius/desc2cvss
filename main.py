import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CONFIG
from cve_dataset import CVEDataset
from cvss_metrics import CVSS_METRICS
from model import SecureBERTWithTFIDF, MultiTaskLoss
from train import evaluate, train_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])

# Создание TF-IDF векторайзера на тренировочных данных
print("Fitting TF-IDF vectorizer...")
data = pandas.read_csv("data/output_ML_all.csv.gz")

train_data, test_data = train_test_split(data, test_size=0.8, random_state=42, stratify=data['year'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['year'])

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

criterion = MultiTaskLoss(num_tasks=len(CVSS_METRICS)).to(DEVICE)

for epoch in range(CONFIG['EPOCHS']):
    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
    print(f"{'=' * 50}")

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    metrics = evaluate(model, val_loader, DEVICE)

    # Вывод результатов
    print("\nValidation Metrics:")
    avg_fbeta = 0
    for metric_name, metric_values in metrics.items():
        print(f"  {metric_name}: Acc={metric_values['accuracy']:.4f}, FBeta={metric_values['fbeta']:.4f}")
        avg_fbeta += metric_values['fbeta']

    avg_fbeta /= len(metrics)
    print(f"  Average FBeta: {avg_fbeta:.4f}")


    path = "cvss_model_epoch_{epoch:02d}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'tfidf_vectorizer': tfidf_vectorizer,
        'metrics': metrics,
    }, path)
    print(f"✓ Model saved to {path}")

print(f"\nTraining complete!")
