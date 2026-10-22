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
  --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts/dmdogo/smoke-01 `
  --status-file artifacts/dmdogo/smoke-01.status.json `
  --stop-after-s 2 `
  --wall-timeout-s 60

start .\artifacts\dmdogo\smoke-01\full-chart-replay.html
```

上記の `--chart` を任意の `.adofai` パスに置き換えられます。
SHA256の固定譜面制限はありません。元譜面のSHA256は結果に自動記録されます。
`--difficulty` は任意の説明ラベルで、**省略時は `unrated`** です。
ファイル名から難易度を推定しません。

`--stop-after-s` を省略すれば譜面全体を処理します（初回は短い区間を推奨）。
出力先とstatusファイルは既存パスを上書きしないため、再実行時は新しい名前にしてください。

## 保存されるもの

- `summary.json`：譜面ハッシュ、Hit/Miss、TooEarly、Overload、X精度、Timing MAE、指使用統計
- `full-chart-replay.json`：各イベントの予定時刻、Network運指、物理KeyDown列、Motor trace
- `full-chart-replay.html`：ブラウザで開けるオフラインリプレイ
- `*.status.json`（`--status-file`で指定）：正常終了・例外の状態

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
