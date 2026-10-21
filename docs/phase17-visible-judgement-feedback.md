# Phase 17 — Visible Judgement Feedback

## 結論

観測系の判定は **`CATEGORY_ANOMALY_OBSERVABLE_IDENTITY_NONIDENTIFIABLE`**
である。可視の累積カウンターだけを受け取る独立
`JudgementObserver`を実装した。productionのTiming、Fingering、Motor、
Scorerおよび既存263D HUDには接続していない。

Phase 16のID 38打鍵では、理想化した10 ms capture上、3.261 sの表示更新を
次の3.270 s frameで`very_late: +1`として取得できる。ID 38自身の予測dueに
対する打鍵誤差は事後評価で-3.706 msなので、独立した因果的な予想判定が
`perfect`なら、観測カテゴリとの不一致は検出できる。

ただし、`very_late: +1`は「正しいイベントを単に遅く叩いた」場合にも同じで
ある。累積値にはtarget IDも同一frame内の順序もない。このためカウンター差分
単独から「Event 37がID 38の打鍵を消費した」と同定することも、安全な補正行動
を選ぶこともできない。

## 確認できた仕様と未確認事項

ユーザーが実機で確認した事実は、画面下部に判定別累積カウンターがあり、判定
発生時に対応値が増えることである。ローカルのDLL監査由来コードには
`HitMargin`相当の分類と`marginTracker.AddHit`履歴がある。一方、現在のリポジトリ
資料だけでは次を確定できない。

- Stock HUDの正確な項目名、分類の集約方法、順序、レイアウト
- 自動MissなどKeyDownを伴わない判定をいつ・どの項目へ表示するか
- 同時判定時の描画順序と、1 frame中の中間状態が見えるか
- 実capture latency、OCR/画像認識誤差、表示更新からcaptureまでの遅延

したがってObserverは`perfect`等を固定enumとして内蔵せず、capture側が明示した
label schemaを扱う。Phase 16診断で使ったlabelはrepository simulatorのKeyDown
margin fixtureであり、Stock HUD項目の確定結果ではない。

## 観測契約

`JudgementCounterFrame`がStudent側の境界であり、含むのは次だけである。

- AIがframeを受け取った`observed_at_s`
- その表示frameの`displayed_at_s`
- 設定済みlabelごとの非負整数の累積値

`JudgementObserver`は次を出力する。

- `BASELINE`: 初回の有効frame。deltaは生成しない。
- `UNCHANGED`: 前回から変化なし。全labelの0 vectorを生成する。
- `DELTA`: 各labelの増分。複数カテゴリと増分個数を保持する。
- `RESET`: 1項目でも減少。負deltaを判定として出さず、新値を基準にする。
- `INVALID`: 欠損・未知label、負数、bool、非因果/非有限時刻。基準を破棄し、
  次の有効frameを新しい`BASELINE`にする。

判定発生時刻は、前回と今回の`displayed_at_s`の区間としてだけ表現する。利用可能
時刻は`observed_at_s`であり、両者を同一視しない。1区間に複数増分があっても、
個々の順序は`within_interval_order_known=False`である。

## HUD-only境界

診断経路は次の順序を強制した。

```text
Phase 16 scorer trace（privileged renderer input）
  -> target_index / key / event identityを破棄
  -> 可視累積counter frame
  -> JudgementObserver
  -> delta vector
  -> 事後EvaluatorでのみPhase 16 truthと比較
```

Observer moduleはScorer、`TimingJudgement`、chart target、`_used`、`_missed`、
`judgement_trace`をimportも保持もしない。型のfield固定テストと、target identityが
異なる2世界でも同じ可視frame/deltaになる反例テストを追加した。

## Phase 16 ID 38の小規模診断

入力は既存のread-only成果物
`reports/phase16-firsttown12-event89-causal-20261021-r2.json`である。

| 項目 | 値 |
| --- | ---: |
| 物理KeyDown | 3.260999999999752 s |
| repository FIFO判定先（事後のみ） | Event 37 |
| 可視margin fixture | `very_late` |
| ID 38予測due（事後比較） | 3.2647059197959036 s |
| dueとの差 | -3.705919796 ms |
| 理想10 ms captureでの初回観測 | 3.270 s |
| 表示発生からの理想観測遅延 | 9.000 ms |

3.260 s frameは差分0、3.270 s frameは`very_late: +1`となった。これは期待
`perfect`とのカテゴリ異常を示せるが、FIFO backlogのevent identityは示さない。
live環境の追加capture latencyは未測定なので、3.270 sは実機保証ではない。

## 変更影響

追加したObserverと監査は独立moduleであり、既存production pathからimportされ
ない。263D/278D/368Dの次元・意味、44 ms gate、Phase 16 Motor A/B、ScorerのFIFO
判定を変更していない。判定状態推定・再同期・追加打鍵・予約キャンセルは未実装
である。

## 検証結果

- Observer＋可視境界監査: 14 passed
- 既存HUD＋Phase 16監査を含む限定回帰: 23 passed
- Timing / Fingering / Motor契約回帰: 109 passed
- repository全体: 1954 passed、既知のPyTorch tensor変換warning 1件

テスト失敗は0件だった。全repository testでもproduction pathの回帰は観測されなかった。

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_judgement_observer_v01.py `
  tests/test_audit_phase17_visible_judgement_v01.py `
  --basetemp artifacts/pytest-phase17-unit

.\.venv\Scripts\python.exe scripts/audit_phase17_visible_judgement_v01.py `
  --source reports/phase16-firsttown12-event89-causal-20261021-r2.json `
  --output reports/phase17-visible-judgement-feedback-20261021-r1.json
```

監査スクリプトは既存出力を上書きしない。`--output`には未作成pathを指定する。
