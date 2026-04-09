import joblib
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from transformers import AutoTokenizer

from config import CONFIG
from cvss_metrics import CVSS_METRICS
from model import SecureBERTWithTFIDF
from train import print_metrics

tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

model = SecureBERTWithTFIDF().to(device)

path = CONFIG['MODEL_PATH_FORMAT'].format(epoch=5)
checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

print(f"Model loaded from {path}\nValidation Metrics:")
print_metrics(checkpoint['metrics'], checkpoint['reduced_metrics'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

config = CONFIG
cvss_metrics = CVSS_METRICS

app = Flask(__name__)

def predict_single(text: str):
    """Предсказание"""
    global model, tfidf_vectorizer, tokenizer, device
    
    # TF-IDF
    tfidf_features = tfidf_vectorizer.transform([text]).toarray()
    tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float32).to(device)
    
    # Токенизация
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=CONFIG['MAX_LEN'],
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Инференс
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, tfidf_tensor)
    
    # Формируем ответ
    logits_dict = {}
    prediction_dict = {}
    
    for metric_name, metric_config in CVSS_METRICS.items():
        logits = outputs[metric_name].cpu().numpy()[0]
        classes = metric_config['classes']
        
        logits_dict[metric_name] = {
            class_name: float(logits[i]) for i, class_name in enumerate(classes)
        }
        
        pred_idx = np.argmax(logits)
        prediction_dict[metric_name] = classes[pred_idx]
    
    return {
        "logits": logits_dict,
        "prediction": prediction_dict
    }

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для предсказания"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        result = predict_single(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/metrics', methods=['GET'])
def metrics_info():
    """Информация о метриках"""
    return jsonify({
        "metrics": list(CVSS_METRICS.keys()),
        "classes_per_metric": {
            metric: config['classes'] for metric, config in CVSS_METRICS.items()
        }
    })


app.run(host='0.0.0.0', port=8000, debug=False)
