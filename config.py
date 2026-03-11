"""設定（パス・モデル名など）"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent

# インデックス・データの保存先
# HuggingFace Spaces では /data が永続ストレージ。それ以外はプロジェクト内の data/ を使用。
_hf_data = Path("/data")
DATA_DIR = _hf_data if _hf_data.exists() else PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
UPLOADS_DIR = DATA_DIR / "uploads"

# Whisper モデル（tiny = 超軽量、base = 軽量、small = バランス、large-v3 = 高精度）
# 無料サーバー(512MB RAM)では base を推奨。ローカルなら small 以上を推奨。
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# ── Gemini API ──────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# 一時的にWhisperのみで動作させる場合は False に
USE_GEMINI: bool = False  # bool(GEMINI_API_KEY)

# Gemini有効時はローカル埋め込みモデル(SentenceTransformer)をスキップ
# → メモリ大幅削減（512MB環境でもOOM回避）
# Gemini無効時のみローカルモデルでベクトル検索
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

GEMINI_CHAT_MODEL = "gemini-2.5-flash"

TOP_K = 3
RERANK_CANDIDATE_K = 10


def ensure_dirs():
    """データ用ディレクトリを自動作成"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
