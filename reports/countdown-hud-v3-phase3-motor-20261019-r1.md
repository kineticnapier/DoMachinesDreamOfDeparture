# HUD v3 Phase 3 — Train bool Motor Integration

## Outcome

Train bool冒頭5秒の固定比較で、HUD v3 CompositeはHUD v1.4より1 Hit改善した。
比較は同一譜面、同一Fingering Network checkpoint、同一12軸Motor、10 ms control、
1 ms physics、同一44/69 ms Leadで1回だけ実施し、閾値探索は行っていない。

| Arm | Hit / Target | Miss | TooEarly | Overload | X Accuracy | Timing MAE | KeyDown |
|---|---:|---:|---:|---:|---:|---:|---:|
| HUD v1.4 | 7 / 9 | 2 | 0 | false | 77.7778% | 2.821886 ms | 7 |
| HUD v3 Composite | 8 / 9 | 1 | 0 | false | 88.8889% | 2.625440 ms | 8 |

差分はHit +1、Miss -1、TooEarly 0、physical KeyDown +1。事前宣言した
`hit_improvement`を含む全GateがPASSした。

## Fixed inputs

- Chart: `04042 - bool - vo[ICE].adofai`
- Chart SHA256: `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`
- Fingering checkpoint: `fingering_network_v01-4fa2ae1.pt`
- Checkpoint SHA256: `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
- Network selector / style alpha: `network` / `0.5`
- Normal / decision Lead: `44 ms` / `69 ms`
- Scope: episode 0–5 s、9 targets

Timing v1.4、Network checkpoint、12軸Motor物理、判定器は変更していない。
TimingEventTokenに既存のoptional uncertaintyを用いたdue下限計算を追加し、
uncertaintyがないv1.4の判断は従来どおりpoint dueになる。

## Motor-before preflight

501 frameのTiming-only監査は全Gate PASS。

- 初回Countdown prediction: step 200、観測2.000 s
- 初回integer Timing ID: `0`（元ID `countdown:0`）
- 初回due interval: `[3.979833385, 3.999833385] s`
- decision-window frame: step 393、3.930 s
- decision時due interval: `[3.989833385, 3.999833385] s`
- midpoint Lead: `64.833385 ms`
- due下限Lead: `59.833385 ms`
- decision時confirmed: true
- pre-due cancellation: 0
- source / interval conflict: 0
- pre-due local ID: 1個のみ
- due intervalの非nested変化: 0

## First-target physical chain

HUD v3では初回ID `0`について次の連鎖を保存ログで確認した。

1. step 393 / 3.930 s: Networkが`left_2`を選択しreservation作成。
2. step 396 / 3.960 s: reservationが`armed_in_flight`へ遷移。
3. step 400内 / 4.001 s: physical `left_2` KeyDown。
4. target 0 due 3.999333624 sに対してerror `+1.666376 ms`、Perfect Hit。

HUD v1.4は初回targetの候補を発行できず、target 0はMiss。step 400で初めて
due 4.020163487 sの次イベント候補を出したが、Leadは約20.16 msしかなく、
`insufficient_observable_lead`として打鍵しなかった。

## Per-target outcome

| Ordinal | Due (s) | v1.4 | v3 | v3 KeyDown / error | v3 contact |
|---:|---:|---|---|---|---|
| 0 | 3.999334 | Miss | Hit | 4.001000 / +1.666 ms | left_2 |
| 1 | 4.020163 | Miss | Miss | — | — |
| 2 | 4.249292 | Hit | Hit | 4.245000 / -4.292 ms | left_1 |
| 3 | 4.270122 | Hit | Hit | 4.273000 / +2.878 ms | left_2 |
| 4 | 4.499250 | Hit | Hit | 4.500000 / +0.750 ms | left_1 |
| 5 | 4.520080 | Hit | Hit | 4.520000 / -0.080 ms | left_2 |
| 6 | 4.749209 | Hit | Hit | 4.742000 / -7.209 ms | left_1 |
| 7 | 4.770039 | Hit | Hit | 4.773000 / +2.961 ms | left_2 |
| 8 | 4.999167 | Hit | Hit | 4.998000 / -1.167 ms | left_1 |

## Failure classification

v3の唯一のMiss（ordinal 1）はTiming層の未対応である。Countdown estimatorの責務は
初回イベントだけで、初回から20.829866 ms後の2イベント目はv1.4へ移行後に初めて
観測された。44 ms Leadを満たさないためFingeringは意図的に未割当とした。

- insufficient Lead: v1.4 `1`、v3 `1`
- unassigned: 両arm `0`
- reservation collision: 両arm `0`
- in-flight cancellation: 両arm `0`
- source conflict: 両arm `0`
- Overload / TooEarly: 両arm `0`

したがって残るMissはNetwork選択、Motor応答、Scorer誤判定ではなく、Countdownが
2イベント目を予測せず、v1.4の観測可能化が遅いことによる。閾値・Lead・物理定数は
変更していない。

## Causality and artifacts

各policyは`TimingFrameUpdate + current Motor`だけを受け取る。AutoPlay target時刻、
`_used/_missed`、judgement traceは全Motor step終了後にのみ照合した。adapterとpolicy
surfaceにはtruth fieldがないことをASTテストしている。

- Full comparison: `artifacts/countdown_hud_v3_phase3/train-bool-20261019-r3/comparison.json`
- Preflight: `reports/countdown-motor-preflight-train-bool-20261019-r1.json`
- stdout/stderr/status: `logs/countdown-hud-v3-phase3-train-bool-20261019-r3.*`

`comparison.json`には全frameのTiming status/pending/canceled、全decision、
reservation/arm/physical edge、target別posthoc結果を保存した。

実験PIDは24832、開始`2026-10-19T00:19:14.5568295+09:00`、終了
`2026-10-19T00:19:24.4443655+09:00`、exit code 0。先行r1/r2はそれぞれPython
launcher/importのCLI起動エラーでMotor frameを実行しておらず、修正後r3だけを結果に
採用した。

## Tests and reproduction

契約・adapter・preflight・既存Timing/Fingering/Motor統合の関連90テスト、および
runner回帰10テストを実行した。最終統合suiteは次のコマンドで再実行する。

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_countdown_hud_v3.py tests/test_countdown_phase_tracker_v01.py tests/test_countdown_first_event_v01.py tests/test_hud_timing_composite_v20.py tests/test_hud_v3_composite_forecast_source_v30.py tests/test_countdown_motor_preflight_v30.py tests/test_timing_fingering_contract_v02.py tests/test_timing_fingering_motor_integration_v02.py tests/test_run_countdown_hud_v3_motor_phase3.py

.\.venv\Scripts\python.exe scripts/audit_countdown_motor_preflight_v30.py --output reports/countdown-motor-preflight-<new-id>.json

.\.venv\Scripts\python.exe scripts/run_countdown_hud_v3_motor_phase3.py --output-dir artifacts/countdown_hud_v3_phase3/<new-id> --wall-timeout-s 180
```

Validation20、FINAL、学習、checkpoint更新は実施していない。この結果はTrain bool冒頭
5秒だけの固定診断であり、未知譜面への一般化を示さない。
