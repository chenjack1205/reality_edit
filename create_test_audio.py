#!/usr/bin/env python3
"""
テスト用の話者別音声ファイルを生成する。
macOS の say コマンドで3人分の日本語音声を作成し、WAV に変換する。

実行: python3 create_test_audio.py
生成先: test_audio/
"""
import subprocess
import sys
from pathlib import Path

# テスト音声の保存先
TEST_DIR = Path(__file__).parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

# 収録日時（ファイル名に使う）→ 自動認識テスト用
RECORD_DATE = "20240601"
RECORD_TIME = "1400"   # 14:00 スタート想定

# 3人のキャラクター設定
# Kyoko ボイスを話速で差別化（155=低め、175=標準、200=速め）
SPEAKERS = [
    {
        "name":  "田中",
        "voice": "Kyoko",
        "rate":  155,  # ゆっくり話す男性キャラ想定
        "lines": [
            "今日のキャンプ、楽しかったな。夕日が本当にきれいで感動した。",
            "佐藤さんと二人きりで話した時、すごく緊張してうまく話せなかった。",
            "正直に言うと、佐藤さんのことが気になってる。どうしたらいいかわからない。",
            "鈴木とは昨日ちょっとケンカしちゃって。俺の言い方が悪かったと思う。反省してる。",
            "この番組に出て、自分が変われた気がする。もっと素直に生きたい。",
        ],
    },
    {
        "name":  "佐藤",
        "voice": "Kyoko",
        "rate":  180,  # 標準速度、女性キャラ想定
        "lines": [
            "田中くんと話せて嬉しかった。なんか不思議な気持ちになった。",
            "本当のことを言うと、田中くんのことが好きかもしれない。でも怖くて言えない。",
            "鈴木くんはいつも明るくて、一緒にいると楽しい。でも友達として見てる。",
            "みんなで花火を見た夜、すごく幸せだった。ずっとこの時間が続けばいいのに。",
            "田中くんがケンカしたって聞いて心配した。仲直りできるといいな。",
        ],
    },
    {
        "name":  "鈴木",
        "voice": "Kyoko",
        "rate":  200,  # 早口、活発な男性キャラ想定
        "lines": [
            "俺、実は佐藤さんのことが好きなんだよね。もう気持ちを抑えられない。",
            "田中には内緒にしてほしいんだけど、そろそろ告白しようと思ってる。",
            "昨日田中と言い合いになってさ。俺も言い方がきつかったって反省してる。",
            "花火の夜、佐藤さんの隣にいられて最高だった。あの瞬間、告白しようかと思った。",
            "最終的には自分の気持ちに正直に生きたい。後悔したくないから。",
        ],
    },
]


def check_dependencies():
    """say と ffmpeg が使えるか確認する。"""
    for cmd in ["say", "ffmpeg"]:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"エラー: '{cmd}' が見つかりません。")
            if cmd == "ffmpeg":
                print("  brew install ffmpeg でインストールしてください。")
            sys.exit(1)
    print("✓ say / ffmpeg ともに利用可能")


def generate_audio(speaker: dict) -> Path:
    """1人分の音声を生成して WAV で返す。"""
    name = speaker["name"]
    voice = speaker["voice"]
    rate = speaker["rate"]
    lines = speaker["lines"]

    # セリフを結合（行間に少し間を空ける）
    script = "　。".join(lines)   # 読点を入れて自然なポーズを作る

    aiff_path = TEST_DIR / f"{name}_temp.aiff"
    wav_path  = TEST_DIR / f"{name}_{RECORD_DATE}_{RECORD_TIME}.wav"

    print(f"  [{name}] 音声生成中 (voice={voice}, rate={rate})...")
    result = subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(aiff_path), script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    エラー: {result.stderr}")
        sys.exit(1)

    print(f"  [{name}] WAV に変換中...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    エラー: {result.stderr}")
        sys.exit(1)

    aiff_path.unlink(missing_ok=True)   # 中間ファイルを削除
    return wav_path


def main():
    print("=== テスト音声ファイルの生成 ===\n")
    check_dependencies()
    print()

    generated = []
    for spk in SPEAKERS:
        wav = generate_audio(spk)
        size_kb = wav.stat().st_size // 1024
        print(f"  ✓ {wav.name}  ({size_kb} KB)\n")
        generated.append(wav)

    print("=" * 45)
    print("✅ 生成完了！\n")
    print("【生成されたファイル】")
    for f in generated:
        print(f"  {f}")
    print()
    print("【ブラウザでのテスト手順】")
    print("1. uvicorn app:app --host 0.0.0.0 --port 8000 でサーバー起動")
    print("2. http://localhost:8000 を開く")
    print(f"3. 「音声ファイルを選択」で test_audio/ 内の3ファイルをまとめて選択")
    print("   → ファイル名から話者名・日時が自動で入ります")
    print("4. 「アップロード」→「文字起こしを開始する」")
    print("5. 完了後、以下のクエリで検索してみてください:")
    print("   ・「佐藤が田中のことが好きと言っているシーン」")
    print("   ・「鈴木が告白しようとしているところ」")
    print("   ・「ケンカをして反省しているシーン」")
    print("   ・「花火を見て幸せだったシーン」")


if __name__ == "__main__":
    main()
