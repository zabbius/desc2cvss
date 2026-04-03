CONFIG = {
    # Модель
    "MODEL_NAME": "cisco-ai/SecureBERT2.0-base",

    # Параметры обучения
    "BATCH_SIZE": 16,
    "EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "MAX_LEN": 512,
    "WARMUP_STEPS": 100,

    # TF-IDF параметры
    "TFIDF_MAX_FEATURES": 512,
    "TFIDF_NGRAM_RANGE": (1, 3),

    # Размерности
    "BERT_HIDDEN_SIZE": 768,  # Размер для SecureBERT/RoBERTa-base
    "TFIDF_FUSION_SIZE": 256,  # Размер после fusion слоя
}
