"""
音声の文字起こし（Whisper）。セグメントごとの開始・終了秒数を返す。
話者ごとの音声ファイルを同じ時間軸（同期済み）で扱う想定。
"""
from pathlib import Path
from dataclasses import dataclass
import gc

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    """1セグメント分の文字起こし（タイムスタンプ付き）"""
    start_sec: float
    end_sec: float
    text: str


def load_whisper(model_size: str = "small", device: str = "auto"):
    """Whisperモデルをロード。device は 'cpu' / 'cuda' / 'auto'。"""
    return WhisperModel(model_size, device=device, compute_type="int8")


def transcribe_file(
    audio_path: Path,
    model: WhisperModel | None = None,
    model_size: str = "small",
    language: str | None = None,
) -> list[TranscriptSegment]:
    """
    1本の音声（または動画）を文字起こしし、セグメントのリストを返す。
    時間軸はファイル先頭を0秒とした相対時間。複数話者で同じ長さ・同期済みなら共通タイムラインとして扱える。
    """
    if model is None:
        model = load_whisper(model_size)
    path_str = str(audio_path.resolve())
    segments_raw, _ = model.transcribe(path_str, language=language, word_timestamps=False)
    segments = [
        TranscriptSegment(
            start_sec=s.start,
            end_sec=s.end,
            text=s.text.strip(),
        )
        for s in segments_raw
        if s.text.strip()
    ]
    return segments


def transcribe_sources(
    sources: list[tuple[Path, str]],
    model_size: str = "small",
    language: str | None = "ja",
) -> list[tuple[str, Path, list[TranscriptSegment]]]:
    """
    話者付き音声を順に文字起こしする。
    sources: (音声ファイルパス, 話者名) のリスト。
    全ファイル処理後にWhisperモデルをメモリから解放する。
    """
    model = load_whisper(model_size)
    results = []
    for path, speaker in sources:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"音声が見つかりません: {path}")
        segments = transcribe_file(path, model=model, language=language)
        results.append((speaker, path, segments))

    del model
    import gc
    gc.collect()

    return results
