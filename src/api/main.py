# src/api/main.py

import os
import re
import joblib
import nltk

from fastapi import FastAPI
from pydantic import BaseModel
from unidecode import unidecode
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_pt = set(stopwords.words('portuguese'))
custom_sw = {'pra','so','ta','to','ja','ai','hoje','vou','vai','mim'}
stop_pt |= custom_sw

# Limpeza de dados
def limpeza_stopwords(text: str) -> str:
    text = text.replace('\n',' ').replace('\r',' ').strip().lower()
    text = unidecode(text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(tok for tok in text.split() if tok not in stop_pt)

# App Fast definido
app = FastAPI(title="Lyrics Genre Classifier")

# Schema de entrada
class LyricsIn(BaseModel):
    lyric: str

# Carrega o pipeline
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(models_dir, 'lyrics_genre_svc.pkl')
pipe = joblib.load(model_path)

# Endpoint
@app.post("/predict")
def predict_genre(data: LyricsIn):
    cleaned = limpeza_stopwords(data.lyric)
    genre = pipe.predict([cleaned])[0]
    return {"genre": genre}
