# DMDOGO v0.1 — マクロ時刻 + 学習済み運指

DMDOGO は DMDOD から派生した小規模なオフライン運指実験です。
正確な打鍵時刻は譜面コンパイラ（AutoPlay Timing Oracle）が与え、
Fingering Network v0.1 が物理部位を選び、既存の12軸・16接点Motorで実行します。

**Timing Student・実機Stock計測Mod・追加学習は使用しません。**
結果はシミュレータ内の入力です。実際のOSキーボード入力やADOFAI本体の自動操作は行いません。

## まず冒頭2秒を実行

PowerShellでリポジトリのルートから実行します。

```powershell
uv run python scripts/run_full_chart_fingering_replay_v02.py `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/03640 - 東 - prisMKa _ Sprout.adofai" `
  --stop-after-s 2
```

上記の `--chart` を任意の `.adofai` パスに置き換えられます。
SHA256の固定譜面制限はありません。元譜面のSHA256は結果に自動記録されます。
`--difficulty` は任意の説明ラベルで、**省略時は `unrated`** です。
ファイル名から難易度を推定しません。

`--stop-after-s` を省略すれば譜面全体を処理します（初回は短い区間を推奨）。
`--checkpoint` は既存の `checkpoints/fingering/fingering_network_v01-4fa2ae1.pt` が既定値です。
`--output-dir` と `--status-file` は省略可能で、それぞれ
`artifacts/dmdogo/<譜面ファイル名>` と `<出力フォルダ>.status.json` が既定値です。

**再実行で出力パスが重複すると、自動で `-002`、`-003`……と両方に付けて保存します。**
既存ファイルを消したり上書きしたりしません。明示指定した出力・statusパスが
存在する場合も同様に連番を付けます。実際に使用した出力パスとHTMLパスは
コンソールに表示されます。

例：`vo[ICE] - bool` の全譜面を実行する場合（64秒の手前で切らないよう
`--stop-after-s` を省略）：

```powershell
uv run python scripts/run_full_chart_fingering_replay_v02.py `
  --chart "C:\Users\Kinetic Napier\Downloads\bool - vo[ICE]\bool\bool.adofai"
```

## 保存されるもの

- `summary.json`：譜面ハッシュ、Hit/Miss、TooEarly、Overload、X精度、Timing MAE、指使用統計
- `full-chart-replay.json`：各イベントの予定時刻、Network運指、物理KeyDown列、Motor trace（v0.2保存形式）
- **`full-chart-replay.html`：標準ビューア v0.3**。16キー表示、KeyDown発光、KPS、打鍵数、物理Motor、惑星軌道、シーク
- `full-chart-replay-v03.json`：v0.3用に描画情報とキーマッピングを追加したJSON
- `full-chart-replay-v02.html`：以前の簡易ビューア（比較用、削除しない）
- `*.status.json`（`--status-file`で指定）：正常終了・例外の状態

v0.3は保存済みReplayを**描画変換するだけ**であり、追加推論、Motorの再実行、学習は行いません。
従来のv0.2リプレイJSON、統計値、判定の記録はそのまま保持します。
過去に作成済みのディレクトリは自動更新しません。必要なら
`scripts/build_fingering_replay_viewer_v03.py` で保存済みJSONを変換してください。

`full-chart-replay.json` の `events[].network` が選ばれた運指を、
`policies.network.keydowns` が実際のシミュレータ上の物理KeyDownを示します。
タイミングはマクロ由来のOracle情報であり、HUD-only Studentの成績ではありません。

## 現段階の境界

- 対応する譜面は `compile_adofai` と既存環境が解釈できる範囲に限ります。
- 特殊イベント・Pause・MultiPlanet等の描画は診断用であり、Unity画面と完全同一ではありません。
- 実機Stockと同じ判定結果は保証しません。
- 学習済みcheckpointを変更しません。
- 新しい学習、実機入力、自動リトライ、FINAL評価、Stock監査は今回の対象外です。

## 最小限の回帰

```powershell
uv run python -m pytest -q tests/test_full_chart_fingering_replay_v02.py
```

既存の保存済みフル譜面実験は
[Full-Chart Fingering Replay v0.2](full-chart-fingering-replay-v02.md)を参照してください。
