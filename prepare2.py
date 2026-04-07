import torch

from config import CONFIG
from model import SecureBERTWithTFIDF
from train import print_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(1, 6):
    model = SecureBERTWithTFIDF().to(DEVICE)
    path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=epoch)
    checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    print(f"Model loaded from {path}\nValidation Metrics:")
    checkpoint['reduced_metrics'] = {
        'weighted_avg_fbeta': checkpoint['weighted_avg_fbeta'],
        'weighted_avg_recall': checkpoint['weighted_avg_recall'],
        'weighted_avg_precision': checkpoint['weighted_avg_precision'],
        'weighted_avg_accuracy': checkpoint['weighted_avg_accuracy'],
    }

    print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])
    model.load_state_dict(checkpoint['model_state_dict'])

    path=f"{path}.n"
    torch.save(model, path)

    torch.save({
        'model_state_dict': checkpoint['model_state_dict'],
        'tfidf_vectorizer': checkpoint['tfidf_vectorizer'],
        'metrics': checkpoint['metrics'],
        'reduced_metrics': checkpoint['reduced_metrics'],
    }, path)
    print(f"✓ Model saved to {path}")

