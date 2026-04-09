import joblib
import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CONFIG
from cve_dataset import CVEDataset
from model import SecureBERTWithTFIDF
from train import evaluate, train_epoch, reduce_metrics, print_metrics

tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

def create_cvss_key(row):
    return "_".join([
        str(row['attack_vector']),
        str(row['user_interaction']),
    ])

data['cvss_key'] = data.apply(create_cvss_key, axis=1)

#train_data, test_data = train_test_split(data, test_size=0.75, random_state=42, stratify=data['cvss_key'])
train_data = data
train_data, val_data = train_test_split(train_data, test_size=0.3, random_state=42, stratify=train_data['cvss_key'])

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])

train_dataset = CVEDataset(train_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)
val_dataset = CVEDataset(val_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)
#test_dataset = CVEDataset(test_data, tokenizer, tfidf_vectorizer=tfidf_vectorizer)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

model = SecureBERTWithTFIDF().to(DEVICE)

print(f"Starting from epoch {CONFIG['START_EPOCH']}")
if CONFIG['START_EPOCH'] > 0:
    path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=CONFIG['START_EPOCH'])
    checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    print(f"Model loaded from {path}\nValidation Metrics:")
    print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])
    model.load_state_dict(checkpoint['model_state_dict'])

# Оптимизатор и scheduler
optimizer = AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])

total_steps = len(train_loader) * CONFIG['EPOCHS']

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=CONFIG['WARMUP_STEPS'], num_training_steps=total_steps)

epoch = CONFIG['START_EPOCH']
endEpoch = CONFIG['START_EPOCH'] + CONFIG['EPOCHS']

# В основном цикле обучения:
while epoch < endEpoch:
    epoch += 1
    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch}/{endEpoch}")
    print(f"{'=' * 50}")

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    metrics = evaluate(model, val_loader, DEVICE)
    reduced_metrics = reduce_metrics(metrics)

    # Вывод результатов с учетом weight каждой метрики
    print("\nValidation Metrics:")
    print_metrics(metrics, reduced_metrics)

    path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'reduced_metrics': reduced_metrics,
    }, path)
    print(f"✓ Model saved to {path}")

print(f"\nTraining complete!")
