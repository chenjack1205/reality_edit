#!/usr/bin/env python3
"""
ブラウザから音声アップロードと検索ができる Web サーバー。
起動: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import re
import shutil
from urllib.parse import quote
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

import config
from src.indexer import build_index
from src.search_engine import search

UPLOADS_MAPPING = config.UPLOADS_DIR / "mapping.json"

_index_status: dict = {"state": "idle", "scenes_count": 0, "error": ""}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    yield


app = FastAPI(title="Reality Edit - シーン検索", lifespan=lifespan)


def _speaker_safe(s: str) -> str:
    s = (s or "").strip() or "speaker"
    return re.sub(r'[\\/:*?"<>|]', "_", s)[:64]


def _parse_filename_meta(filename: str) -> dict:
    """
    ファイル名から話者名と収録開始日時を自動抽出する。
    対応パターン（拡張子除く）:
      YYYYMMDD_HHMM_話者名      例: 20240115_1430_田中
      YYYYMMDD_HHMMSS_話者名    例: 20240115_143045_田中
      YYYYMMDD_HHMMSSCC_話者名  例: 20240115_14302557_田中
      話者名_YYYYMMDD_HHMM      例: 田中_20240115_1430
      話者名_YYYYMMDD_HHMMSS    例: 田中_20240115_143045
      話者名_YYYYMMDD_HHMMSSCC  例: 田中_20240115_14302557
    どのパターンにも合わない場合は { speaker: ファイル名ベース, file_start_iso: "" } を返す。
    """
    stem = Path(filename).stem

    def _build_iso(date_str: str, time_str: str) -> str:
        try:
            if len(time_str) == 4:   # HHMM → HHMMSS00
                time_str += "0000"
            elif len(time_str) == 6: # HHMMSS → HHMMSSCC(=00)
                time_str += "00"
            # time_str は HHMMSSCC (8桁) として処理
            h, mi, s, cs = time_str[:2], time_str[2:4], time_str[4:6], time_str[6:8]
            return (
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                f"T{h}:{mi}:{s}.{cs}"
            )
        except Exception:
            return ""

    # YYYYMMDD_HHMM(SS)(CC)_話者名
    m = re.match(r"^(\d{8})_(\d{4}|\d{6}|\d{8})_(.+)$", stem)
    if m:
        date_str, time_str, speaker = m.groups()
        return {"speaker": speaker, "file_start_iso": _build_iso(date_str, time_str)}

    # 話者名_YYYYMMDD_HHMM(SS)(CC)
    m = re.match(r"^(.+)_(\d{8})_(\d{4}|\d{6}|\d{8})$", stem)
    if m:
        speaker, date_str, time_str = m.groups()
        return {"speaker": speaker, "file_start_iso": _build_iso(date_str, time_str)}

    return {"speaker": stem, "file_start_iso": ""}


@app.get("/", response_class=HTMLResponse)
def index_page():
    path = Path(__file__).parent / "static" / "index.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="static/index.html not found")
    return FileResponse(path)


@app.post("/parse-filename")
def parse_filename(filename: str = Form(...)):
    """ファイル名から話者名と収録開始日時を抽出して返す（UI の自動入力用）。"""
    return _parse_filename_meta(filename)


@app.post("/upload")
def upload_audios(
    files: list[UploadFile] = File(...),
    metadata: str = Form("[]"),
):
    """
    音声ファイルをアップロードする。
    metadata: JSON配列文字列。各要素: { "speaker": "田中", "file_start_iso": "2024-01-15T14:30:00" }
              ファイルの順番と一致させること。
    """
    if not files:
        raise HTTPException(status_code=400, detail="ファイルを1つ以上選択してください")

    try:
        meta_list: list[dict] = json.loads(metadata)
    except Exception:
        raise HTTPException(status_code=400, detail="metadata が正しい JSON 形式ではありません")

    if len(meta_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"メタデータの数（{len(meta_list)}）とファイル数（{len(files)}）が一致しません",
        )

    config.ensure_dirs()
    for f in config.UPLOADS_DIR.iterdir():
        if f.is_file():
            f.unlink()

    mapping = []
    for i, uf in enumerate(files):
        m = meta_list[i]
        speaker = (m.get("speaker") or "").strip() or _parse_filename_meta(uf.filename or "").get("speaker", f"speaker_{i}")
        file_start_iso = (m.get("file_start_iso") or "").strip()

        ext = Path(uf.filename or "audio").suffix or ".wav"
        # 元ファイル名をそのまま使う（特殊文字だけ除去）。
        # 同じ話者の複数ファイルを上書きしないよう、話者名ではなく元名を維持する。
        orig_stem = re.sub(r'[\\/:*?"<>|]', "_", Path(uf.filename or f"audio_{i}").stem)[:120]
        dest_name = f"{orig_stem}{ext}"
        dest = config.UPLOADS_DIR / dest_name
        with dest.open("wb") as out:
            shutil.copyfileobj(uf.file, out)
        mapping.append({
            "path": dest_name,
            "speaker": speaker,
            "file_start_iso": file_start_iso,
        })

    UPLOADS_MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(files), "mapping": mapping}


def _run_index_task():
    global _index_status
    _index_status = {"state": "running", "scenes_count": 0, "error": ""}
    try:
        mapping = json.loads(UPLOADS_MAPPING.read_text(encoding="utf-8"))
        # sources: (path, speaker, file_start_iso)
        sources = [
            (config.UPLOADS_DIR / m["path"], m["speaker"], m.get("file_start_iso", ""))
            for m in mapping
        ]
        scenes = build_index(sources, whisper_model_size=config.WHISPER_MODEL, language="ja")
        _index_status = {"state": "done", "scenes_count": len(scenes), "error": ""}
    except Exception as e:
        _index_status = {"state": "error", "scenes_count": 0, "error": str(e)}


@app.post("/index")
def run_index(background_tasks: BackgroundTasks):
    global _index_status
    if not UPLOADS_MAPPING.exists():
        raise HTTPException(status_code=400, detail="先に音声をアップロードしてください")
    mapping = json.loads(UPLOADS_MAPPING.read_text(encoding="utf-8"))
    sources = [(config.UPLOADS_DIR / m["path"], m["speaker"], m.get("file_start_iso", "")) for m in mapping]
    for p, _, _ in sources:
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"アップロードファイルが見つかりません: {p.name}")
    if _index_status.get("state") == "running":
        raise HTTPException(status_code=409, detail="インデックス作成が既に実行中です")
    background_tasks.add_task(_run_index_task)
    return {"ok": True, "state": "running"}


@app.get("/index/status")
def index_status():
    return _index_status


@app.get("/search")
def search_api(q: str = "", top_k: int = 3):
    if not q.strip():
        return {"hits": []}
    hits = search(q.strip(), top_k=top_k)
    return {
        "hits": [
            {
                "speaker": h.speaker,
                "start_sec": h.start_sec,
                "end_sec": h.end_sec,
                "abs_start_iso": h.abs_start_iso,
                "abs_end_iso": h.abs_end_iso,
                "score": h.score,
                "text": h.text,
                "reason": h.reason,
            }
            for h in hits
        ],
        "gemini": config.USE_GEMINI,
    }


@app.get("/transcripts")
def list_transcripts():
    """文字起こし済みファイルの一覧を返す。"""
    files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            total_sec = data[-1]["end_sec"] if data else 0
            result.append({
                "speaker": f.stem,
                "segments": len(data),
                "duration_sec": round(total_sec, 1),
                "filename": f.name,
            })
        except Exception:
            result.append({"speaker": f.stem, "segments": 0, "duration_sec": 0, "filename": f.name})
    return {"files": result}


@app.get("/transcripts/download")
def download_all_transcripts():
    """全話者の文字起こしを1つの読みやすいテキストファイルとして返す。"""
    files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="文字起こしデータがまだありません")

    lines = []
    for f in files:
        speaker = f.stem
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        lines.append(f"{'='*60}")
        lines.append(f"【{speaker}】")
        lines.append(f"{'='*60}")
        for seg in data:
            start = _sec_to_hms(seg["start_sec"])
            end   = _sec_to_hms(seg["end_sec"])
            lines.append(f"[{start} → {end}]  {seg['text']}")
        lines.append("")

    text = "\n".join(lines)
    return PlainTextResponse(
        content=text,
        headers={"Content-Disposition": 'attachment; filename="transcripts.txt"'},
        media_type="text/plain; charset=utf-8",
    )


@app.get("/transcripts/download/{speaker}")
def download_speaker_transcript(speaker: str):
    """指定した話者の文字起こしをテキストファイルとして返す。"""
    path = config.TRANSCRIPTS_DIR / f"{speaker}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"話者 {speaker} の文字起こしが見つかりません")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="ファイルの読み込みに失敗しました")

    lines = [f"【{speaker}】文字起こし", ""]
    for seg in data:
        start = _sec_to_hms(seg["start_sec"])
        end   = _sec_to_hms(seg["end_sec"])
        lines.append(f"[{start} → {end}]  {seg['text']}")

    text = "\n".join(lines)
    fname = f"transcript_{speaker}.txt"
    return PlainTextResponse(
        content=text,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"},
        media_type="text/plain; charset=utf-8",
    )


def _sec_to_hms(sec: float) -> str:
    """秒数を HH:MM:SS.CC 形式に変換。"""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
