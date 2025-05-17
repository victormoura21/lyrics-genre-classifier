# src/models/train_and_save.py

import os
import joblib
import pandas as pd
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

# stopwords

nltk.download('stopwords', quiet=True)
stop_pt = set(stopwords.words('portuguese'))
custom_sw = {'pra','so','ta','to','ja','ai','hoje','vou','vai','mim'}
stop_pt |= custom_sw

# limpeza de dados
def limpeza_stopwords(text: str) -> str:
    text = text.replace('\n',' ').replace('\r',' ').strip().lower()
    text = unidecode(text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(tok for tok in text.split() if tok not in stop_pt)

# Importa o pipeline 
from src.pipelines.multimodal import pipe_svc

def load_dataset():
    # Encontra a pasta data/raw na raiz do projeto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_dir = os.path.join(project_root, 'data', 'raw')

    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith('.csv'):
            genre = os.path.splitext(fname)[0]
            df = pd.read_csv(os.path.join(raw_dir, fname))
            df['genre'] = genre

            df['cleaned'] = df['lyric'].astype(str).apply(limpeza_stopwords)

            dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    return full['cleaned'], full['genre']

def main():
    # Carrega e limpa
    X, y = load_dataset()

    # Treina o pipeline no dataset completo
    pipe_svc.fit(X, y)

    # Serializa o modelo treinado
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, 'lyrics_genre_svc.pkl')
    joblib.dump(pipe_svc, out_path)
    print(f"Modelo treinado e salvo em: {out_path}")

if __name__ == "__main__":
    main()