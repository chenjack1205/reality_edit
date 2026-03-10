"""
文字起こし結果をシーンに分割し、誰が何を言っているかをインデックスしてベクトルDBに保存する。
各音声ファイルに「収録開始日時（file_start_iso）」を付与することで、
シーンの絶対日時（何日の何時何分）を記録できる。
"""
import gc
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import config
from src.transcribe import transcribe_sources, TranscriptSegment


def _speaker_to_filename(speaker: str) -> str:
    """話者名を安全なファイル名に変換。"""
    s = re.sub(r'[\\/:*?"<>|]', "_", speaker.strip()) or "speaker"
    return s[:64]


def segments_to_scenes(
    speaker: str,
    segments: list[TranscriptSegment],
    file_start_iso: str = "",
    merge_short_sec: float = 5.0,
) -> list[dict]:
    """
    セグメントを「シーン」にまとめる。
    file_start_iso があれば、各シーンに絶対日時（abs_start_iso / abs_end_iso）を付与する。
    例: file_start_iso="2024-01-15T14:30:00", start_sec=120.5 → abs_start_iso="2024-01-15T14:32:00.5"
    """
    file_dt: datetime | None = None
    if file_start_iso:
        try:
            file_dt = datetime.fromisoformat(file_start_iso)
        except ValueError:
            file_dt = None

    scenes = []
    current_texts: list[str] = []
    start_sec: float | None = None
    end_sec: float | None = None

    for s in segments:
        if start_sec is None:
            start_sec = s.start_sec
        end_sec = s.end_sec
        current_texts.append(s.text)

        duration = end_sec - start_sec
        if duration >= merge_short_sec or len("".join(current_texts)) > 200:
            scene = _make_scene(speaker, start_sec, end_sec, current_texts, file_dt)
            scenes.append(scene)
            current_texts = []
            start_sec = None

    if current_texts and start_sec is not None:
        scene = _make_scene(speaker, start_sec, end_sec, current_texts, file_dt)
        scenes.append(scene)

    return scenes


def _to_cs_iso(dt: datetime) -> str:
    """datetime を 0.01秒（センチ秒）精度の ISO 文字列に変換。例: 2024-01-15T14:32:00.57"""
    cs = dt.microsecond // 10000  # マイクロ秒 → センチ秒（0〜99）
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{cs:02d}"


def _make_scene(
    speaker: str,
    start_sec: float,
    end_sec: float,
    texts: list[str],
    file_dt: datetime | None,
) -> dict:
    scene: dict = {
        "speaker": speaker,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "text": " ".join(texts).strip(),
        "abs_start_iso": "",
        "abs_end_iso": "",
    }
    if file_dt is not None:
        scene["abs_start_iso"] = _to_cs_iso(file_dt + timedelta(seconds=start_sec))
        scene["abs_end_iso"] = _to_cs_iso(file_dt + timedelta(seconds=end_sec))
    return scene


def build_index(
    sources: list[tuple[Path, str, str]],
    whisper_model_size: str = "small",
    language: str | None = "ja",
    output_transcripts_dir: Path | None = None,
) -> list[dict]:
    """
    話者付き音声を文字起こし → シーン分割 → 埋め込み → Chroma に保存。
    sources: (音声ファイルパス, 話者名, 収録開始ISO文字列) のリスト。
             収録開始ISO文字列 例: "2024-01-15T14:30:00"（不明なら空文字）
    """
    config.ensure_dirs()
    out_dir = output_transcripts_dir or config.TRANSCRIPTS_DIR

    # 前回の文字起こしJSONを削除（古いデータが残らないように）
    for old_file in out_dir.glob("*.json"):
        try:
            old_file.unlink()
        except Exception:
            pass

    # transcribe_sources は (path, speaker) のみ受け取るので変換
    transcribe_input = [(p, spk) for p, spk, _ in sources]
    # file_start_iso は speaker でルックアップできるよう辞書化
    start_iso_map = {spk: iso for _, spk, iso in sources}

    print(f"[indexer] Whisper文字起こし開始（モデル: {whisper_model_size}）", flush=True)
    transcription_results = transcribe_sources(
        transcribe_input, model_size=whisper_model_size, language=language
    )
    print("[indexer] Whisper文字起こし完了 → メモリ解放済み", flush=True)

    all_scenes: list[dict] = []
    for speaker, path, segments in transcription_results:
        file_start_iso = start_iso_map.get(speaker, "")
        scenes = segments_to_scenes(speaker, segments, file_start_iso=file_start_iso)
        all_scenes.extend(scenes)
        print(f"[indexer]   {speaker}: {len(segments)} セグメント → {len(scenes)} シーン", flush=True)

        transcript_path = out_dir / f"{_speaker_to_filename(speaker)}.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"start_sec": s.start_sec, "end_sec": s.end_sec, "text": s.text} for s in segments],
                f,
                ensure_ascii=False,
                indent=2,
            )

    gc.collect()

    if not all_scenes:
        return all_scenes

    # シーンJSONは常に保存（Geminiモードでもローカルモードでも使う）
    scenes_path = config.INDEX_DIR / "scenes.json"
    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"[indexer] scenes.json 保存完了: {len(all_scenes)} シーン")

    if config.USE_GEMINI:
        # Gemini有効時: ローカル埋め込みモデル不要（メモリ節約）
        # 検索時にGeminiが直接シーンを評価する
        print("[indexer] Geminiモード: ローカル埋め込みスキップ（メモリ節約）")
    else:
        # Gemini無効時: ローカルモデルで埋め込み → ChromaDB
        print("[indexer] ローカルモード: 埋め込みベクトル計算開始")
        from src.embeddings import embed_texts
        texts = [s["text"] for s in all_scenes]
        embeddings = embed_texts(texts, task="retrieval_document")
        print("[indexer] 埋め込み完了 → ChromaDB保存")

        import chromadb
        index_path = config.INDEX_DIR / "chroma"
        client = chromadb.PersistentClient(path=str(index_path))
        try:
            client.delete_collection("scenes")
        except Exception:
            pass
        collection = client.create_collection("scenes", metadata={"description": "reality show scenes"})

        ids = [f"{s['speaker']}_{s['abs_start_iso'] or s['start_sec']}" for s in all_scenes]
        metadatas = [
            {
                "speaker": s["speaker"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "abs_start_iso": s["abs_start_iso"],
                "abs_end_iso": s["abs_end_iso"],
                "text": s["text"][:500],
            }
            for s in all_scenes
        ]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print("[indexer] ChromaDB保存完了")

    return all_scenes
