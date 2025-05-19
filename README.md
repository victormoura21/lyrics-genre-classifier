# Lyrics Genre Classifier

Classificador de gêneros musicais (Bossa Nova, Funk, Gospel, Sertanejo) usando TF‑IDF + SVM.

## Estrutura do Repositório

```
lyrics-genre-classifier/
├── data/raw/              # CSVs brutos de letras por gênero
├── notebooks/             # Análise exploratória (EDA)
├── src/
│   ├── pipelines/         # Definição dos pipelines de ML
│   │   └── multimodal.py  # TF‑IDF + LinearSVC (Modelo final)
│   ├── models/            # Scripts e artefatos de modelo
│   │   ├── train_and_save.py   # Treina e serializa o pipeline
│   │   └── lyrics_genre_svc.pkl # Modelo treinado serializado
│   ├── api/               # Serviço REST com FastAPI
│   │   └── main.py        # Endpoints e pré-processamento (CORS ativado)
│   └── client/            # Front-end HTML + JS para testar localmente
│       └── index.html     # Interface simples de classificação
├── .gitignore
├── requirements.txt
└── README.md              # Este arquivo
```

## Como Rodar

### 1. Instalação

```bash
# Crie e ative um virtual environment
python -m venv .venv                # ou python3 -m venv .venv
. .\.venv\Scripts\Activate.ps1        # Windows PowerShell

# Instale as dependências
pip install -r requirements.txt
```

### 2. Treinar e Serializar o Modelo

```bash
# Na raiz do projeto:
python -m src.models.train_and_save
```

> Isso vai treinar o pipeline com todo o dataset em `data/raw` e gerar `src/models/lyrics_genre_svc.pkl`.

### 3. Iniciar o Serviço REST (FastAPI)

```bash
# Ainda na raiz:
uvicorn src.api.main:app --reload
```

- Acesse a interface interativa em: `http://127.0.0.1:8000/docs`
- Endpoint principal: **POST /predict**
  - **Request JSON**: `{ "lyric": "texto da música..." }`
  - **Response JSON**: `{ "genre": "bossa_nova" }`

### 4. Testar o Front-end

1. Abra `src/client/index.html` no navegador (duplo clique).   
2. Cole a letra no textarea e clique em **Classificar**.  
3. O gênero previsto aparece na página.

> **Obs:** o CORS já está habilitado no servidor para permitir requests do `file://`
