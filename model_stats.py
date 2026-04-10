import torch

from config import CONFIG
from train import print_metrics

for epoch in range(1, 8):
    path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=epoch)
    checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

    print(f"Model loaded from {path}\nValidation Metrics:")
    print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])

    print(f"\n{'-' * 50}")
