#!/usr/bin/env python3
"""
リアリティショー素材のシーン検索 CLI。
話者ごとの音声を同じ時間軸で文字起こしし、「誰が何秒で何を言っているか」で検索する。

使い方:
  # 話者付き音声をインデックス（path:話者名 の形式）
  python main.py index --sources "path/to/田中.wav:田中" "path/to/佐藤.wav:佐藤"

  # クエリでトップ3を表示（誰が・何秒で・何を言ったか）
  python main.py search "誰がどういったことを言っているところか"
"""
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config
from src.indexer import build_index
from src.search_engine import search, SceneHit

console = Console()


def _parse_sources(sources_raw: list[str], audios: list[str] | None, speakers: list[str] | None) -> list[tuple[Path, str]]:
    """--sources または --audios + --speakers から (Path, 話者名) のリストを組み立てる。"""
    if sources_raw:
        out = []
        for s in sources_raw:
            if ":" in s:
                path_str, speaker = s.rsplit(":", 1)
                out.append((Path(path_str.strip()), speaker.strip()))
            else:
                p = Path(s.strip())
                out.append((p, p.stem))
        return out
    if audios and speakers:
        if len(audios) != len(speakers):
            raise ValueError("--audios と --speakers の数は同じにしてください")
        return [(Path(a), sp) for a, sp in zip(audios, speakers)]
    raise ValueError("--sources か、--audios と --speakers の両方を指定してください")


def cmd_index(args):
    config.ensure_dirs()
    try:
        sources = _parse_sources(
            getattr(args, "sources", None) or [],
            getattr(args, "audios", None),
            getattr(args, "speakers", None),
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    for p, _ in sources:
        if not p.exists():
            console.print(f"[red]ファイルが見つかりません: {p}[/red]")
            return 1
    console.print(f"[cyan]音声の文字起こしとインデックス作成を開始します（{len(sources)} 話者）...[/cyan]")
    scenes = build_index(sources, whisper_model_size=args.model, language=args.language or None)
    console.print(f"[green]完了。シーン数: {len(scenes)}[/green]")
    return 0


def cmd_search(query: str, top_k: int):
    hits = search(query, top_k=top_k)
    if not hits:
        console.print("[yellow]インデックスがありません。先に 'index' で音声を登録してください。[/yellow]")
        return 1

    table = Table(title=f"「{query}」に近いシーン Top{top_k}")
    table.add_column("順位", style="dim")
    table.add_column("話者", style="cyan")
    table.add_column("開始(秒)", justify="right")
    table.add_column("終了(秒)", justify="right")
    table.add_column("スコア", justify="right")
    table.add_column("発言内容（抜粋）", max_width=50)

    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i),
            h.speaker,
            str(h.start_sec),
            str(h.end_sec),
            f"{h.score:.2%}",
            (h.text[:80] + "…") if len(h.text) > 80 else h.text,
        )
    console.print(table)
    console.print("\n[bold]該当箇所（誰の音声の何秒で言っているか）:[/bold]")
    for i, h in enumerate(hits, 1):
        snippet = (h.text[:60] + "…") if len(h.text) > 60 else h.text
        console.print(f"  {i}. [cyan]{h.speaker}[/cyan] の [green]{h.start_sec}秒〜{h.end_sec}秒[/green] で 「{snippet}」 と言っています。")
    return 0


def main():
    parser = argparse.ArgumentParser(description="リアリティショー素材のシーン検索（話者・時間軸付き音声）")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="話者付き音声を文字起こししてインデックスを作成")
    p_index.add_argument("--sources", nargs="+", default=[], help='"パス:話者名" のリスト（例: "person1.wav:田中" "person2.wav:佐藤"）')
    p_index.add_argument("--audios", nargs="+", help="音声ファイルのパス（--speakers とセット）")
    p_index.add_argument("--speakers", nargs="+", help="話者名のリスト（--audios と同順・同数）")
    p_index.add_argument("--model", default="small", choices=["base", "small", "medium", "large"], help="Whisper モデル")
    p_index.add_argument("--language", default="ja", help="言語コード（例: ja）")

    # search
    p_search = sub.add_parser("search", help="クエリでシーンを検索（誰が何秒で何を言ったか）")
    p_search.add_argument("query", help="例: 誰がどういったことを言っているところか")
    p_search.add_argument("--top", type=int, default=3, help="表示する件数（デフォルト 3）")

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    if args.command == "search":
        return cmd_search(args.query, top_k=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
