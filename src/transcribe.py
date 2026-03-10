"""
音声の文字起こし。
Gemini API が利用可能なら Gemini で高精度に文字起こし（ローカルMLモデル不要）。
そうでなければ faster-whisper を使用。
"""
import gc
import json
from pathlib import Path
from dataclasses import dataclass

import config


@dataclass
class TranscriptSegment:
    """1セグメント分の文字起こし（タイムスタンプ付き）"""
    start_sec: float
    end_sec: float
    text: str


# ── Gemini 文字起こし ───────────────────────────────────────────────────────

_MIME_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}


def _transcribe_file_gemini(
    audio_path: Path,
    language: str = "ja",
) -> list[TranscriptSegment]:
    """Gemini API で音声を文字起こし。タイムスタンプ付きJSON形式で返す。"""
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    import shutil, tempfile
    mime_type = _MIME_MAP.get(audio_path.suffix.lower(), "audio/mpeg")
    print(f"[transcribe] Gemini: アップロード中 {audio_path.name} ({mime_type})", flush=True)

    # Gemini APIはファイル名にASCII以外を受け付けないので一時ファイルで回避
    suffix = audio_path.suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(audio_path, tmp_path)
    try:
        uploaded = client.files.upload(file=tmp_path, config={"mime_type": mime_type})
    finally:
        tmp_path.unlink(missing_ok=True)

    prompt = f"""この音声ファイルを正確に文字起こししてください。
言語: {"日本語" if language == "ja" else language}

以下のJSON配列形式のみで返してください（他の文章は不要）：
[
  {{"start_sec": 0.0, "end_sec": 2.5, "text": "発話内容"}},
  {{"start_sec": 2.5, "end_sec": 5.0, "text": "次の発話"}},
  ...
]

ルール:
- start_sec / end_sec は音声先頭からの秒数（小数点以下2桁まで）
- 聞こえた言葉をそのまま正確に書き起こす（意訳しない）
- 無音区間はスキップ
- セグメントは発話単位で分割（長すぎないように）
"""

    print(f"[transcribe] Gemini: 文字起こし実行中 {audio_path.name}", flush=True)
    response = client.models.generate_content(
        model=config.GEMINI_CHAT_MODEL,
        contents=[uploaded, prompt],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    segments_data = json.loads(response.text)
    segments = [
        TranscriptSegment(
            start_sec=float(s.get("start_sec", 0)),
            end_sec=float(s.get("end_sec", 0)),
            text=s.get("text", "").strip(),
        )
        for s in segments_data
        if s.get("text", "").strip()
    ]
    print(f"[transcribe] Gemini: {audio_path.name} → {len(segments)} セグメント", flush=True)
    return segments


# ── Whisper 文字起こし（フォールバック）─────────────────────────────────────

def _load_whisper(model_size: str = "small"):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _transcribe_file_whisper(
    audio_path: Path,
    model=None,
    model_size: str = "small",
    language: str | None = None,
) -> list[TranscriptSegment]:
    if model is None:
        model = _load_whisper(model_size)
    path_str = str(audio_path.resolve())
    segments_raw, _ = model.transcribe(path_str, language=language, word_timestamps=False)
    return [
        TranscriptSegment(start_sec=s.start, end_sec=s.end, text=s.text.strip())
        for s in segments_raw
        if s.text.strip()
    ]


# ── メインエントリポイント ──────────────────────────────────────────────────

def transcribe_sources(
    sources: list[tuple[Path, str]],
    model_size: str = "small",
    language: str | None = "ja",
) -> list[tuple[str, Path, list[TranscriptSegment]]]:
    """
    話者付き音声を順に文字起こしする。
    Gemini 有効時: Gemini API で高精度文字起こし（ローカルモデル不要）。
    Gemini 無効時: faster-whisper で文字起こし（処理後にメモリ解放）。
    """
    results = []

    if config.USE_GEMINI:
        print("[transcribe] Geminiモードで文字起こし開始", flush=True)
        for path, speaker in sources:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"音声が見つかりません: {path}")
            try:
                segments = _transcribe_file_gemini(path, language=language or "ja")
            except Exception as e:
                print(f"[transcribe] Gemini失敗、Whisperにフォールバック: {e}", flush=True)
                model = _load_whisper(model_size)
                segments = _transcribe_file_whisper(path, model=model, language=language)
                del model
                gc.collect()
            results.append((speaker, path, segments))
    else:
        print("[transcribe] Whisperモードで文字起こし開始", flush=True)
        model = _load_whisper(model_size)
        for path, speaker in sources:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"音声が見つかりません: {path}")
            segments = _transcribe_file_whisper(path, model=model, language=language)
            results.append((speaker, path, segments))
        del model
        gc.collect()

    return results
