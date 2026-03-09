"""
自然言語クエリでシーンを検索し、「誰が何日何時何分で何を言っているか」を返す。
GEMINI_API_KEY が設定されている場合:
  1. Gemini 埋め込みで上位 RERANK_CANDIDATE_K 件を候補として取得
  2. Gemini Flash がクエリとの関連度を判定し、最終 TOP_K 件に絞り込む
未設定の場合:
  ローカルモデルによるベクトル検索のみ（再ランキングなし）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

import config
from src.embeddings import embed_texts


@dataclass
class SceneHit:
    """検索でヒットした1シーン（誰が・いつ・何を言ったか）"""
    speaker: str
    start_sec: float
    end_sec: float
    score: float
    text: str
    abs_start_iso: str = field(default="")
    abs_end_iso: str = field(default="")
    reason: str = field(default="")  # Gemini 再ランキング時の判定理由


def search(
    query: str,
    top_k: int | None = None,
    index_dir: Path | None = None,
) -> list[SceneHit]:
    """
    クエリで意味的に近いシーンを返す。
    Gemini が有効なら候補を多めに取得して再ランキングする。
    """
    top_k = top_k or config.TOP_K
    index_path = index_dir or config.INDEX_DIR / "chroma"

    if not index_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(index_path))
    try:
        collection = client.get_collection("scenes")
    except Exception:
        return []

    # Gemini 有効時は候補を多めに取得して再ランキング
    candidate_k = config.RERANK_CANDIDATE_K if config.USE_GEMINI else top_k
    candidate_k = min(candidate_k, collection.count())
    if candidate_k == 0:
        return []

    query_embedding = embed_texts([query], task="retrieval_query")

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        include=["metadatas", "distances"],
    )

    candidates: list[SceneHit] = []
    if results["ids"] and results["ids"][0]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            candidates.append(SceneHit(
                speaker=meta["speaker"],
                start_sec=meta["start_sec"],
                end_sec=meta["end_sec"],
                score=round(score, 4),
                text=meta.get("text", ""),
                abs_start_iso=meta.get("abs_start_iso", ""),
                abs_end_iso=meta.get("abs_end_iso", ""),
            ))

    if not candidates:
        return []

    if config.USE_GEMINI:
        return _rerank_with_gemini(query, candidates, top_k)

    return candidates[:top_k]


# ── Gemini 再ランキング ───────────────────────────────────────────────────────

def _rerank_with_gemini(query: str, candidates: list[SceneHit], top_k: int) -> list[SceneHit]:
    """
    Gemini Flash に候補リストとクエリを渡し、最も関連するシーンを選んで返す。
    失敗した場合はベクトルスコア順のまま返す。
    """
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        candidates_text = ""
        for i, h in enumerate(candidates):
            time_info = h.abs_start_iso if h.abs_start_iso else f"音声の{h.start_sec}秒目"
            candidates_text += f"[{i}] {h.speaker}（{time_info}）：「{h.text[:300]}」\n"

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

        reranked: list[SceneHit] = []
        for pos, idx in enumerate(ranking[:top_k]):
            if 0 <= idx < len(candidates):
                hit = candidates[idx]
                hit.reason = reasons[pos] if pos < len(reasons) else ""
                reranked.append(hit)

        return reranked if reranked else candidates[:top_k]

    except Exception as e:
        # 再ランキング失敗時はベクトルスコア順をそのまま返す
        print(f"[rerank] Gemini 再ランキング失敗（フォールバック）: {e}")
        return candidates[:top_k]
