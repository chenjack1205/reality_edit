FROM python:3.11-slim

# ffmpeg（音声変換に必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU専用PyTorchをインストール（CUDA版を避けてイメージサイズ削減）
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 残りの依存パッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体をコピー
COPY . .

# データ用ディレクトリを作成
RUN mkdir -p data/uploads data/index data/transcripts

# Whisperモデルをプリダウンロード（Gemini失敗時のフォールバック用）
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8'); print('Whisper tiny downloaded')"

EXPOSE 8000
CMD ["python", "app.py"]
