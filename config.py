"""設定（パス・モデル名など）"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent

# インデックス・データの保存先
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
UPLOADS_DIR = DATA_DIR / "uploads"

# Whisper モデル（tiny = 超軽量、base = 軽量、small = バランス、large-v3 = 高精度）
# 無料サーバー(512MB RAM)では base を推奨。ローカルなら small 以上を推奨。
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ── Gemini API（検索の再ランキングのみ）──────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# APIキーがあれば検索結果の再ランキングに Gemini を使う
# ベクトル化・インデックスは常にローカルモデルを使用
USE_GEMINI: bool = bool(GEMINI_API_KEY)

# 埋め込みモデル（常にローカル）
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 再ランキング用 Gemini チャットモデル
GEMINI_CHAT_MODEL = "gemini-2.0-flash"

# 検索で返す件数（Gemini 再ランキング時は候補を多めに取って絞り込む）
TOP_K = 3
RERANK_CANDIDATE_K = 10  # Gemini 再ランキング前の候補数


def ensure_dirs():
    """データ用ディレクトリを自動作成"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
