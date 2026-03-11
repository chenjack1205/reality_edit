"""
音声の文字起こし。
Gemini API が利用可能なら Gemini で高精度に文字起こし（ローカルMLモデル不要）。
長い音声は ffmpeg で5分チャンクに分割してGeminiに渡し、タイムスタンプを合成する。
そうでなければ faster-whisper を使用。
"""
import gc
import json
import re
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


def _cap_segment_duration(seg: TranscriptSegment) -> TranscriptSegment:
    """Geminiの過大なend_secを補正。日本語は約4〜5文字/秒程度なので、テキスト長から妥当な最大長を算出"""
    duration = seg.end_sec - seg.start_sec
    # 4文字/秒を基準に最大長を算出（最低2秒、最高25秒）
    max_duration = min(25.0, max(2.0, len(seg.text) / 4.0))
    if duration > max_duration:
        return TranscriptSegment(
            start_sec=seg.start_sec,
            end_sec=round(seg.start_sec + max_duration, 2),
            text=seg.text,
        )
    return seg


def _normalize_text(text: str) -> str:
    """Geminiの分かち書き（単語間スペース）を除去して自然な日本語にする"""
    # 日本語文字（ひらがな・カタカナ・漢字）の間のスペースを削除
    text = re.sub(
        r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff])\s+([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff])",
        r"\1\2",
        text,
    )
    # 連続スペースを1つに
    text = re.sub(r"\s+", " ", text).strip()
    return text


class QuotaExceededError(Exception):
    """Gemini APIのquota/レート制限超過"""


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

# 1チャンクの長さ（秒）。長いほどリクエスト数減・無料枠消費を抑える
_CHUNK_SEC = 900  # 15分


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
    各チャンクを -ss START -t DURATION で正確に切り出し。
    offset は元ファイルの実際の開始秒と一致する。
    """
    duration = _get_duration(audio_path)
    if duration <= 0:
        return [(audio_path, 0.0)]

    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        chunk_dur = min(chunk_sec, duration - start)
        chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
        # -ss を -i の前に置くと入力シーク（高速）。-t で正確な長さを指定
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(audio_path),
            "-t", str(chunk_dur),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append((chunk_path, start))
        else:
            print(f"[transcribe] チャンク分割失敗 start={start}s", flush=True)
        start += chunk_sec
        idx += 1

    if not chunks:
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
- セグメントは発話単位で分割（1文や1フレーズごと。細かく切りすぎない）
- セグメントは時系列順に並べ、重複やオーバーラップを避ける（前のend_secと次のstart_secは連続または無音区間を挟む）
- 同じ内容を複数回出力しない
- 日本語の場合は分かち書きをせず、通常の文章として書く（単語間にスペースを入れない）
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
    segments = []
    for s in segments_data:
        raw = s.get("text", "").strip()
        if not raw:
            continue
        text = _normalize_text(raw)
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_sec=round(float(s.get("start_sec", 0)) + offset_sec, 2),
                end_sec=round(float(s.get("end_sec", 0)) + offset_sec, 2),
                text=text,
            )
        )
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
                    # このチャンクの範囲内のセグメントのみ採用（Geminiの誤ったタイムスタンプを除去）
                    chunk_end = (
                        chunks[idx + 1][1] if idx + 1 < len(chunks) else duration
                    )
                    for s in segs:
                        if offset_sec <= s.start_sec < chunk_end and s.start_sec < s.end_sec:
                            all_segments.append(s)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    err_repr = repr(e)
                    is_quota = (
                        "429" in err_str
                        or "429" in err_repr
                        or "RESOURCE_EXHAUSTED" in err_str.upper()
                        or "QUOTA" in err_str.upper()
                        or "EXCEEDED" in err_str.upper()
                        or getattr(e, "code", None) == 429
                        or getattr(e, "status_code", None) == 429
                    )
                    if is_quota:
                        print(
                            "[transcribe] Gemini quota超過 → Whisperに自動切り替え",
                            flush=True,
                        )
                        raise QuotaExceededError("Gemini API quota exceeded") from e
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

    # 時系列ソート（start_sec, end_sec の順で安定ソート）
    all_segments.sort(key=lambda s: (s.start_sec, s.end_sec))

    # 完全同一の重複のみ除去（同じ発言を2回した場合は時間が違うので両方残る）
    # テキスト・時間が完全に同じセグメントだけ破棄（Geminiの二重出力対策）
    seen: set[tuple[float, float, str]] = set()
    final: list[TranscriptSegment] = []
    for s in all_segments:
        s = _cap_segment_duration(s)  # Geminiの過大なend_secを補正
        key = (round(s.start_sec, 1), round(s.end_sec, 1), s.text)
        if key in seen:
            continue
        seen.add(key)
        final.append(s)

    final.sort(key=lambda s: (s.start_sec, s.end_sec))
    print(
        f"[transcribe] Gemini: {audio_path.name} 合計 {len(final)} セグメント",
        flush=True,
    )
    return final


# ── Gemini + Whisper マージ（テキストはGemini、時間はWhisper）────────────────────

def _align_with_whisper_timestamps(
    gemini_segments: list[TranscriptSegment],
    whisper_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """
    Geminiのテキスト + Whisperのタイムスタンプでマージ。
    テキスト量の比率で対応付け、Whisperの正確な時間をGeminiの各セグメントに割り当てる。
    """
    if not gemini_segments or not whisper_segments:
        return gemini_segments

    total_g = sum(len(g.text) for g in gemini_segments)
    total_w = sum(len(w.text) for w in whisper_segments)
    if total_g == 0 or total_w == 0:
        return gemini_segments

    result: list[TranscriptSegment] = []
    w_idx = 0
    n = len(gemini_segments)

    for i, g in enumerate(gemini_segments):
        is_last = i == n - 1
        target_share = len(g.text) / total_g
        target_w_len = target_share * total_w

        seg_start: float | None = None
        seg_end: float | None = None
        seg_acc = 0.0

        while w_idx < len(whisper_segments) and (is_last or seg_acc < target_w_len):
            w = whisper_segments[w_idx]
            if seg_start is None:
                seg_start = w.start_sec
            seg_end = w.end_sec
            seg_acc += len(w.text)
            w_idx += 1
            if not is_last and seg_acc >= target_w_len:
                break

        if seg_start is not None and seg_end is not None:
            result.append(
                TranscriptSegment(start_sec=seg_start, end_sec=seg_end, text=g.text)
            )
        else:
            result.append(g)

    return result


# ── Whisper 文字起こし（フォールバック・タイムスタンプ補正）──────────────────────

def _load_whisper(model_size: str = "tiny"):
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
        TranscriptSegment(
            start_sec=s.start,
            end_sec=s.end,
            text=_normalize_text(s.text.strip()),
        )
        for s in segments_raw
        if s.text.strip()
    ]


# ── メインエントリポイント ──────────────────────────────────────────────────

def transcribe_sources(
    sources: list[tuple[Path, str]],
    model_size: str = "small",
    language: str | None = "ja",
    fallback_messages: list[str] | None = None,
) -> list[tuple[str, Path, list[TranscriptSegment]]]:
    """
    話者付き音声を順に文字起こしする。
    Gemini 有効時: Gemini API で高精度文字起こし（長音声は5分チャンク分割）。
    Gemini 無効時: faster-whisper で文字起こし。
    """
    results = []

    if config.USE_GEMINI:
        print("[transcribe] Geminiモードで文字起こし開始（Whisper tinyでタイムスタンプ補正）", flush=True)
        model = _load_whisper(model_size)
        try:
            for path, speaker in sources:
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(f"音声が見つかりません: {path}")
                segments = []
                try:
                    segments = _transcribe_file_gemini(path, language=language or "ja")
                except QuotaExceededError:
                    pass
                except Exception as e:
                    print(f"[transcribe] Gemini失敗、Whisperにフォールバック: {e}", flush=True)
                if not segments:
                    print(
                        f"[transcribe] {path.name}: セグメント0件（quota超過等）→ Whisperにフォールバック",
                        flush=True,
                    )
                    if fallback_messages is not None and len(fallback_messages) == 0:
                        fallback_messages.append(
                            "GeminiのAPI制限のため、Whisperで文字起こししました。"
                        )
                    segments = _transcribe_file_whisper(path, model=model, language=language)
                else:
                    print(f"[transcribe] {path.name}: Whisper tinyでタイムスタンプ補正中", flush=True)
                    whisper_segs = _transcribe_file_whisper(path, model=model, language=language)
                    segments = _align_with_whisper_timestamps(segments, whisper_segs)
                results.append((speaker, path, segments))
        finally:
            del model
            gc.collect()
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
