import os
import joblib

from src.pipelines.multimodal import pipe_svc

def main():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    out_path = os.path.join(models_dir, 'lyrics_genre_svc.pkl')
    joblib.dump(pipe_svc, out_path)
    print(f"Modelo salvo em: {out_path}")

if __name__ == "__main__":
    main()