"""
音声の文字起こし。
Gemini API が利用可能なら Gemini で高精度に文字起こし（ローカルMLモデル不要）。
長い音声は ffmpeg で5分チャンクに分割してGeminiに渡し、タイムスタンプを合成する。
そうでなければ faster-whisper を使用。
"""
import gc
import json
import shutil
import subprocess
import tempfile
import time
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

# 1チャンクの長さ（秒）。長いほどリクエスト数減・レート制限に有利
_CHUNK_SEC = 300  # 5分


def _get_duration(audio_path: Path) -> float:
    """ffprobeで音声の長さ（秒）を取得する。"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def _split_audio(audio_path: Path, chunk_sec: int, tmp_dir: Path) -> list[tuple[Path, float]]:
    """
    ffmpeg segment muxer で音声を chunk_sec 秒ごとに一括分割。
    (ファイルパス, 開始秒) のリストを返す。
    segment muxer はキーフレームではなく指定秒数で正確に切るため、
    タイムスタンプのずれが最小限に抑えられる。
    """
    duration = _get_duration(audio_path)
    if duration <= 0:
        return [(audio_path, 0.0)]

    out_pattern = str(tmp_dir / "chunk_%04d.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ac", "1",           # モノラル
        "-ar", "16000",       # 16kHz
        "-sample_fmt", "s16",
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        "-reset_timestamps", "1",   # 各チャンクのタイムスタンプを0始まりにリセット
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)

    chunks = []
    idx = 0
    while True:
        chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            break
        offset = idx * chunk_sec
        chunks.append((chunk_path, float(offset)))
        idx += 1

    if not chunks:
        print(f"[transcribe] segment分割失敗: {result.stderr.decode()[-300:]}", flush=True)
        return [(audio_path, 0.0)]

    print(f"[transcribe] {len(chunks)} チャンクに分割完了", flush=True)
    return chunks


def _upload_and_transcribe_chunk(
    client,
    chunk_path: Path,
    offset_sec: float,
    mime_type: str,
    language: str,
    file_label: str,
    chunk_idx: int,
    total_chunks: int,
) -> list[TranscriptSegment]:
    """1チャンクをGeminiにアップロードして文字起こし。タイムスタンプにoffset_secを加算して返す。"""
    from google.genai import types as gtypes

    # 一時ファイル（ASCII名）でアップロード
    with tempfile.NamedTemporaryFile(suffix=chunk_path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(chunk_path, tmp_path)

    try:
        print(
            f"[transcribe] Gemini: {file_label} チャンク{chunk_idx+1}/{total_chunks} "
            f"({offset_sec/60:.1f}分〜) アップロード中",
            flush=True,
        )
        uploaded = client.files.upload(file=tmp_path, config={"mime_type": mime_type})
    finally:
        tmp_path.unlink(missing_ok=True)

    # ファイルが ACTIVE になるまで待機（最大60秒）
    for _ in range(30):
        file_info = client.files.get(name=uploaded.name)
        state = getattr(file_info, "state", None)
        if state is None or str(state) in ("ACTIVE", "FileState.ACTIVE", "2"):
            break
        if str(state) in ("FAILED", "FileState.FAILED", "3"):
            raise RuntimeError(f"Geminiファイル処理失敗: {uploaded.name}")
        time.sleep(2)
    print(
        f"[transcribe] Gemini: チャンク{chunk_idx+1}/{total_chunks} 文字起こし実行中",
        flush=True,
    )

    prompt = f"""この音声クリップを正確に文字起こししてください。
言語: {"日本語" if language == "ja" else language}

以下のJSON配列形式のみで返してください（他の文章は不要）：
[
  {{"start_sec": 0.0, "end_sec": 2.5, "text": "発話内容"}},
  {{"start_sec": 2.5, "end_sec": 5.0, "text": "次の発話"}},
  ...
]

ルール:
- start_sec / end_sec はこのクリップ先頭からの秒数（0.0始まり）
- 聞こえた言葉をそのまま正確に書き起こす（意訳しない）
- 無音区間はスキップ
- セグメントは発話単位で分割
"""

    response = client.models.generate_content(
        model=config.GEMINI_CHAT_MODEL,
        contents=[uploaded, prompt],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=16384,
        ),
    )

    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    segments_data = json.loads(response.text)
    segments = [
        TranscriptSegment(
            start_sec=round(float(s.get("start_sec", 0)) + offset_sec, 2),
            end_sec=round(float(s.get("end_sec", 0)) + offset_sec, 2),
            text=s.get("text", "").strip(),
        )
        for s in segments_data
        if s.get("text", "").strip()
    ]
    print(
        f"[transcribe] Gemini: チャンク{chunk_idx+1}/{total_chunks} → {len(segments)} セグメント",
        flush=True,
    )
    return segments


def _transcribe_file_gemini(
    audio_path: Path,
    language: str = "ja",
) -> list[TranscriptSegment]:
    """Gemini API で音声を文字起こし。長い音声は5分チャンクに分割して処理する。"""
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    mime_type = _MIME_MAP.get(audio_path.suffix.lower(), "audio/mpeg")

    duration = _get_duration(audio_path)
    print(
        f"[transcribe] Gemini: {audio_path.name} 長さ={duration/60:.1f}分 "
        f"→ {max(1, int(duration/_CHUNK_SEC)+1)} チャンクで処理",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks = _split_audio(audio_path, _CHUNK_SEC, Path(tmp_dir))
        all_segments: list[TranscriptSegment] = []
        for idx, (chunk_path, offset_sec) in enumerate(chunks):
            # チャンクはWAVに変換済みなのでMIMEタイプを上書き
            chunk_mime = _MIME_MAP.get(chunk_path.suffix.lower(), mime_type)
            # チャンク間インターバル（レート制限対策: 10 RPM = 6秒/リクエスト）
            if idx > 0:
                time.sleep(7)

            # 最大2回リトライ
            last_err = None
            for attempt in range(2):
                try:
                    segs = _upload_and_transcribe_chunk(
                        client=client,
                        chunk_path=chunk_path,
                        offset_sec=offset_sec,
                        mime_type=chunk_mime,
                        language=language,
                        file_label=audio_path.name,
                        chunk_idx=idx,
                        total_chunks=len(chunks),
                    )
                    all_segments.extend(segs)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"[transcribe] チャンク{idx+1} 試行{attempt+1}/2 失敗: {e}",
                        flush=True,
                    )
                    if attempt == 0:
                        time.sleep(15)  # リトライ前に少し待つ
            if last_err is not None:
                print(
                    f"[transcribe] チャンク{idx+1}/{len(chunks)} リトライ後も失敗 "
                    f"(offset={offset_sec:.0f}s): {last_err}",
                    flush=True,
                )

    # チャンク間でタイムスタンプが逆行する場合があるので時系列に並び替え
    all_segments.sort(key=lambda s: s.start_sec)

    print(
        f"[transcribe] Gemini: {audio_path.name} 合計 {len(all_segments)} セグメント",
        flush=True,
    )
    return all_segments


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
    Gemini 有効時: Gemini API で高精度文字起こし（長音声は5分チャンク分割）。
    Gemini 無効時: faster-whisper で文字起こし。
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
