"""
自然言語クエリでシーンを検索する。

Gemini 有効時:
  scenes.json を直接読み、Gemini にクエリとの関連度を判定させる。
  ローカル埋め込みモデル不要 → メモリ大幅節約。

Gemini 無効時:
  ローカルモデルで埋め込み → ChromaDB ベクトル検索。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class SceneHit:
    speaker: str
    start_sec: float
    end_sec: float
    score: float
    text: str
    abs_start_iso: str = field(default="")
    abs_end_iso: str = field(default="")
    reason: str = field(default="")


def search(
    query: str,
    top_k: int | None = None,
    index_dir: Path | None = None,
) -> list[SceneHit]:
    top_k = top_k or config.TOP_K
    index_dir = index_dir or config.INDEX_DIR

    if config.USE_GEMINI:
        return _search_with_gemini(query, top_k, index_dir)
    else:
        return _search_with_chroma(query, top_k, index_dir)


# ── Gemini 直接検索 ─────────────────────────────────────────────────────────

def _search_with_gemini(query: str, top_k: int, index_dir: Path) -> list[SceneHit]:
    scenes_path = index_dir / "scenes.json"
    if not scenes_path.exists():
        return []

    try:
        all_scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not all_scenes:
        return []

    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    candidates_text = ""
    for i, s in enumerate(all_scenes):
        time_info = s.get("abs_start_iso") or f"音声の{s['start_sec']}秒目"
        candidates_text += f"[{i}] {s['speaker']}（{time_info}）：「{s['text'][:300]}」\n"

    prompt = f"""あなたはリアリティショーの映像編集者です。
以下の音声文字起こしの候補シーンの中から、検索クエリに最も合うシーンを選んでください。

検索クエリ：「{query}」

候補シーン（番号は0始まり）：
{candidates_text}

上位 {top_k} 件を選び、以下の JSON 形式のみで回答してください（他の文章は不要）：
{{
  "ranking": [最も関連が高い候補番号, 2番目, 3番目, ...],
  "reasons": ["1位の選んだ理由（20字以内）", "2位の理由", "3位の理由", ...]
}}
"""

    try:
        response = client.models.generate_content(
            model=config.GEMINI_CHAT_MODEL,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)
        ranking: list[int] = result.get("ranking", [])
        reasons: list[str] = result.get("reasons", [])

        hits: list[SceneHit] = []
        for pos, idx in enumerate(ranking[:top_k]):
            if 0 <= idx < len(all_scenes):
                s = all_scenes[idx]
                hits.append(SceneHit(
                    speaker=s["speaker"],
                    start_sec=s["start_sec"],
                    end_sec=s["end_sec"],
                    score=round(1.0 - pos * 0.1, 4),
                    text=s["text"],
                    abs_start_iso=s.get("abs_start_iso", ""),
                    abs_end_iso=s.get("abs_end_iso", ""),
                    reason=reasons[pos] if pos < len(reasons) else "",
                ))
        return hits if hits else []

    except Exception as e:
        print(f"[search] Gemini検索失敗: {e}")
        return _fallback_keyword_search(query, all_scenes, top_k)


def _fallback_keyword_search(
    query: str, scenes: list[dict], top_k: int
) -> list[SceneHit]:
    """Gemini失敗時のフォールバック: 単純なキーワードマッチ"""
    query_chars = set(query)
    scored = []
    for s in scenes:
        text = s["text"]
        overlap = len(query_chars & set(text))
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = []
    for score, s in scored[:top_k]:
        hits.append(SceneHit(
            speaker=s["speaker"],
            start_sec=s["start_sec"],
            end_sec=s["end_sec"],
            score=round(score / max(len(query_chars), 1), 4),
            text=s["text"],
            abs_start_iso=s.get("abs_start_iso", ""),
            abs_end_iso=s.get("abs_end_iso", ""),
            reason="キーワードマッチ（フォールバック）",
        ))
    return hits


# ── ローカル ChromaDB 検索（Gemini無効時）──────────────────────────────────────

def _search_with_chroma(query: str, top_k: int, index_dir: Path) -> list[SceneHit]:
    import chromadb
    from src.embeddings import embed_texts

    index_path = index_dir / "chroma"
    if not index_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(index_path))
    try:
        collection = client.get_collection("scenes")
    except Exception:
        return []

    candidate_k = min(top_k, collection.count())
    if candidate_k == 0:
        return []

    query_embedding = embed_texts([query], task="retrieval_query")

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        include=["metadatas", "distances"],
    )

    hits: list[SceneHit] = []
    if results["ids"] and results["ids"][0]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            hits.append(SceneHit(
                speaker=meta["speaker"],
                start_sec=meta["start_sec"],
                end_sec=meta["end_sec"],
                score=round(score, 4),
                text=meta.get("text", ""),
                abs_start_iso=meta.get("abs_start_iso", ""),
                abs_end_iso=meta.get("abs_end_iso", ""),
            ))

    return hits
