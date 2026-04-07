import joblib
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CONFIG

data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

tfidf_vectorizer = TfidfVectorizer(
    max_features=CONFIG['TFIDF_MAX_FEATURES'],
    ngram_range=CONFIG['TFIDF_NGRAM_RANGE'],
    stop_words='english'
)

X = tfidf_vectorizer.fit_transform(data['description'])

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')