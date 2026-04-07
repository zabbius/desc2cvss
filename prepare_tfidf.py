import joblib
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


data = pandas.read_csv("data/filtered_output_ML_all.csv.gz")

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['description'])

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')