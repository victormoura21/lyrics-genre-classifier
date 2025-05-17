import os
import joblib
import pandas as pd

from src.pipelines.multimodal import pipe_svc

def load_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_dir = os.path.join(project_root, 'data', 'raw')

    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith('.csv'):
            genre = os.path.splitext(fname)[0]
            df = pd.read_csv(os.path.join(raw_dir, fname))
            df['genre'] = genre
            dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full['cleaned'], full['genre']

def main():
    X, y = load_dataset()

    pipe_svc.fit(X, y)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, 'lyrics_genre_svc.pkl')
    joblib.dump(pipe_svc, out_path)
    print(f"Modelo treinado e salvo em: {out_path}")

if __name__ == "__main__":
    main()