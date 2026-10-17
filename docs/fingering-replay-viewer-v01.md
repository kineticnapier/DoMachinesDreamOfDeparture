# Real-Chart Fingering Replay Viewer v0.1

## 結果

既存のreal-chart pilot 4区間について、譜面、診断用惑星、12物理軸Motor、
Teacher v0.2 / Fingering Network v0.1の固定済み運指、実KeyDown、判定を単一の
episode時間軸で同期再生するself-contained HTML viewerを生成した。

新規学習・再推論・checkpoint更新は行っていない。保存済みpilot decisionを固定入力として
Motor状態だけを決定論的にreplayし、既存JSONのHit/Miss/TooEarly/Overload/X/Timing MAEと
完全一致しない場合は生成を中止する。

## 操作

- 再生 / 一時停止
- seek bar
- 0.25x / 0.5x / 1x / 2x
- 1 event前 / 次
- event番号へのjump
- Teacher / Network表示切替

seek後の表示は履歴の差分適用ではなく、指定episode時刻からfloor、event、Motor trace、
KeyDownを毎回binary searchして再構成する。そのため通常再生とランダムアクセスで同じ状態になる。

12軸は左右各4指、親指、手側面を中心に表示する。緑枠は押下、橙枠は復帰、枠なしは待機。
現在eventにはTeacher/Networkの採用部位、差、Hit/Miss/TooEarly、実KeyDown timing errorを表示する。

## 時間軸

exportには次の値と式を明示保存する。

```text
chart_s = segment_start_chart_s + episode_s * pitch_ratio
audio_s = (chart_s + offset_ms / 1000 - separate_countdown_chart_s) / pitch_ratio
```

床の`target_time_s`、`exit_time_s`、Pause、SetSpeed、Twirl、Midspin、BPM、回転方向、
惑星数は`CompiledAdoFaiChart`からexportする。pilot 4区間にはTwirl 59、SetSpeed 7、
Pause 1、BPM transition 7が含まれ、Midspinは0だった。Midspin・Pause・SetSpeed・Twirlと
pitch/separate-countdown変換は合成fixtureのunit testで別途検証した。

## 描画上の制約

- gameplay floor geometryとcompiled timingだけを描画する。
- 装飾、filter、camera event、Unity easingは未対応。
- 惑星移動はPauseを保持した診断用曲線補間で、公式Unity rendererの再現ではない。
- 保存済み物理評価値はHTML内で再計算せず、そのままpolicy別に表示する。
- 物理axisを主表示とし、logical key名は中心表示に使わない。

## 検証

関連test: **18 passed**。

- one-chart pilot: 1/1生成、fixed-decision replay一致
- final: 4/4生成、87 events
- floor index / event floor対応: PASS
- episode / chart / audio round trip: PASS
- seek direct lookup vs sequential trace index: 32/32一致
- Teacher / Network replay metrics: 全項目一致
- embedded JavaScript `node --check`: 4/4 PASS
- 外部CDN / server dependency: なし
- viewer出力: write-once、既存pilot viewerを上書きしない

ブラウザ自動操作では`file://` navigationがbrowser security policyで拒否されたため、
自動クリックによる最終目視検証は実施できなかった。生成HTMLはローカルで直接開ける。

## 成果物

- index: `artifacts/fingering_replay_v01/pilot-2437517-r2/index.html`
- manifest: `artifacts/fingering_replay_v01/pilot-2437517-r2/manifest.json`
- 各区間: 同directoryの`replay-*.html` / `replay-*.json`
- status: `logs/fingering-replay-v01-2437517-r2.status.json`
- log: `logs/fingering-replay-v01-2437517-r2.log`

manifest SHA256:
`c6e16594a069bae04bbdce517e4411ee55bc85fd704335e19c6b1d2de6a4abca`

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_fingering_replay_viewer_v01.py `
  tests/test_fingering_real_chart_v02.py `
  tests/test_fingering_network_v01.py `
  tests/test_oracle_fingering_v02.py -q `
  --basetemp artifacts/pytest-fingering-replay-v01-repro

.\.venv\Scripts\python.exe -m scripts.build_fingering_replay_viewer_v01 `
  --report reports/fingering-real-chart-v02-pilot-010fc19-r4.json `
  --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts/fingering_replay_v01/pilot-repro `
  --status-file logs/fingering-replay-v01-repro.status.json `
  *> logs/fingering-replay-v01-repro.log
```

Stage A FAIL、MaleCNS4096、Fingering Network v0.1 checkpoint、元pilot JSONは読み取り専用で、
開始時のSHA256から変更されていない。Qualification20 / Reserve20 / Finalは開封していない。
