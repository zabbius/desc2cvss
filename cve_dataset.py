from sklearn.externals.array_api_compat import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

from config import CONFIG
from cvss_metrics import CVSS_METRICS

class CVEDataset(Dataset):
    def __init__(self, data, tokenizer, tfidf_vectorizer=None):

        self.data = data
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer

        texts = self.data['description']

        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=CONFIG['TFIDF_MAX_FEATURES'],
                ngram_range=CONFIG['TFIDF_NGRAM_RANGE'],
                stop_words='english'
            )
            self.tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            self.tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        description = item['description']

        # Токенизация
        encoding = self.tokenizer(
            description,
            truncation=True,
            padding='max_length',
            max_length=CONFIG['MAX_LEN'],
            return_tensors='pt'
        )

        # TF-IDF признаки
        tfidf = torch.tensor(self.tfidf_features[idx], dtype=torch.float32)

        # Метки для всех метрик
        labels = {}
        for metric_name in CVSS_METRICS.keys():
            labels[metric_name] = torch.tensor(
                item[metric_name],
                dtype=torch.long
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tfidf_features': tfidf,
            'labels': labels
        }
