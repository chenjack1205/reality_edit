"""
テキスト埋め込みレイヤー。
ベクトル化は常にローカルの SentenceTransformers を使用。
Gemini は search_engine.py の再ランキングのみで使用。
"""
from __future__ import annotations

import config


def embed_texts(texts: list[str], task: str = "retrieval_document") -> list[list[float]]:
    """テキストリストをローカルモデルでベクトル化して返す。使用後にモデルを解放。"""
    if not texts:
        return []
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    result = model.encode(texts).tolist()
    del model
    import gc
    gc.collect()
    return result
