"""
テキスト埋め込みレイヤー。
ベクトル化は常にローカルの SentenceTransformers を使用。
Gemini は search_engine.py の再ランキングのみで使用。
"""
from __future__ import annotations

import config

_local_model = None


def embed_texts(texts: list[str], task: str = "retrieval_document") -> list[list[float]]:
    """テキストリストをローカルモデルでベクトル化して返す。"""
    if not texts:
        return []
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_model.encode(texts).tolist()
