# src/pipelines/multimodal.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# Stopwrods + Custo,
stop_pt = set(stopwords.words('portuguese'))
custom_sw = {'pra', 'so', 'ta', 'to', 'ja', 'ai', 'hoje', 'vou', 'vai', 'mim'}
stop_pt |= custom_sw

# Pipeline
pipe_svc = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            min_df=2,
            stop_words=list(stop_pt),
        ),
    ),
    (
        "clf",
        LinearSVC(
            class_weight="balanced",
            max_iter=5_000,
            random_state=42,
        ),
    ),
])
