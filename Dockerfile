FROM python:3.11-slim

# ffmpeg（音声変換に必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU専用PyTorchを先にインストール（CUDA版を避けることで ~2GB → ~250MB に削減）
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 残りの依存パッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体をコピー
COPY . .

# データ用ディレクトリを作成
RUN mkdir -p data/uploads data/index data/transcripts

EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
