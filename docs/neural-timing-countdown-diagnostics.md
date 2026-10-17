# DMDOD Neural Timing Countdown Diagnostics

## 結論

Timing Student v3 は、同じイベントに対して一貫した countdown を生成できていない。
固定した 120/240/480 BPM の単純譜面では、閾値適用前の first-slot 時刻回帰が
イベントまでの残り時間ではなく、およそ 40--50 ms の定常値に留まった。
したがって v4 では時間的一貫性を明示的に学習させるべきである。ただし、tracker
だけを変更しても raw 回帰の失敗は解消しないため、時刻 head と存在 head の学習目標を
先に修正し、小規模 fixture で gate を通してから未知譜面へ進むべきである。

## 固定条件と因果性

- checkpoint: `checkpoints/prestige1/autoplay_sequence_student_v3-e07d934.pt`
- checkpoint SHA256:
  `bca41284d85e92c6a9be2c0a2b134201609310d8e2a534f1c7f518900c30ca1a`
- checkpoint format: `prestige1-autoplay-multievent-timing-student-v3-stable-time`
- confidence threshold: 0.5
- fixture: 180-degree ordinary floors, 12 events each, 120/240/480 BPM
- simulator: 10 ms control, 1 ms physics, idle 8-key Motor
- HUD: genuine 263D HUD v2; v3 と Timing v1.4 は同じ frame 列を使用
- Motor は全キー 0 のため、AutoPlay 真値による先行 action cue は HUD に入らない
- v3 推論は current/past HUD の forward-only GRU のみを使用
- v3 と v1.4 の全 frame 推論および causal tracker trace を完了した後にのみ、
  evaluator が `segment.targets` の正解時刻を materialize した
- 正解時刻は event association、誤差、failure attribution にのみ使用した
- Qualification20 / Reserve20 は未使用。大規模再学習・threshold 探索は未実施

この fixture は未知譜面 holdout の代用ではなく、countdown 挙動を分解するための
決定論的 unit-level diagnostic である。

## 集計結果

36 events、288 event-window frames（各イベントの 0--80 ms 前）を評価した。

| method | presence coverage | raw due MAE | raw countdown step MAE | monotonic violations | disappear / reappear | tracker ID switches |
|---|---:|---:|---:|---:|---:|---:|
| Student v3 | 26.74% | 21.360 ms | 9.896 ms | 45 | 0 / 0 | 25 |
| Timing v1.4 | 89.58% | 0.0000076 ms | 0.00000028 ms | 0 | 30 / 0 | 0 |

`raw countdown step MAE` は、連続 frame 間で
`predicted_remaining(t+dt) - predicted_remaining(t) + dt` の絶対値を測る。
理想的な 10 ms countdown は 0 ms になる。v3 の約 9.9 ms は、予測残り時間が
ほぼ減少していないことを示す。

Timing v1.4 の disappearance はイベント境界直前の floating-point 上ほぼ 0 ms の
frame で予測を消す挙動であり、10--80 ms の準備区間では正しい countdown を維持した。
再出現は初回 acquisition を除外し、いったん出現した同一 event が消えた後に再度
出現した場合だけを数えた。

### BPM 別

| BPM | v3 presence | v3 due MAE | v3 step MAE | v3 monotonic violations | v3 ID switches | v1.4 due MAE | v1.4 step MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 0.00% | 22.333 ms | 9.997 ms | 35 | 0 | 0.0000138 ms | 0.00000028 ms |
| 240 | 22.92% | 21.686 ms | 9.928 ms | 3 | 6 | 0.0000063 ms | 0.00000028 ms |
| 480 | 57.29% | 20.060 ms | 9.762 ms | 7 | 19 | 0.0000026 ms | 0.00000028 ms |

v3 の event-window mean confidence の event median は 120/240/480 BPM でそれぞれ
0.411 / 0.463 / 0.531 だった。速度により confidence が上がる一方、時刻精度は
改善しておらず、confidence は countdown correctness を表していない。

## 代表的な同一イベント列

120 BPM の一例では真値が 70, 60, 50, 40, 30, 20, 10 ms と減る間、v3 は
50.121, 50.120, 50.122, 50.126, 50.133, 50.142, 50.153 ms を出した。
confidence は 0.334 から 0.337 で、全 frame が threshold 未満だった。
Timing v1.4 は同じ HUD で 70, 60, 50, 40, 30, 20, 10 ms を再現した。

480 BPM の一例では真値 75, 65, 55, 45, 35, 25, 15, 5 ms に対し、v3 は
42.939, 43.895, 44.569, 45.024, 45.326, 45.531, 45.688, 45.836 ms だった。
残り時間が減るべき区間でむしろ増えている。

## Failure attribution

1. **時刻回帰: 主原因。** threshold 前から due MAE 21.36 ms、step MAE
   9.90 ms であり、存在判定や tracker を介さなくても countdown が成立しない。
2. **イベント存在判定: 独立した失敗。** coverage 26.74%。特に 120 BPM は 0% で、
   raw first-slot が存在しても confidence が threshold を越えない。
3. **予測再対応付け: 観測された下流症状。** 10 truth events が 20 ms 以内で複数
   local ID に分断され、25 switches があった。ただし raw time regression が既に
   failure criterion を超えるため、独立した根因とは判定しない。v1.4 は 0 switches。

## v4 の判断

v4 では時間的一貫性を学習させるべきである。少なくとも同一 truth event に対して、
教師側だけで event identity を使い、Student 入力には identity や未来情報を渡さずに、
次を小規模 train-heldout fixture で検証する。

- absolute due invariance: `observation_time + predicted_remaining` が同一 event 内で一定
- countdown slope: 10 ms 後の remaining が約 10 ms 減る pairwise loss
- presence persistence: event が horizon に入った後の confidence の不要な点滅を抑制
- absolute timing loss は維持し、「傾きだけ正しく常に offset」の解を許さない
- raw head の fixture gate を先に設け、due MAE <= 10 ms、step MAE <= 2 ms、
  presence coverage >= 80% を満たすまで tracker tuning や未知譜面評価へ進まない

これは v4 の大規模学習を直ちに正当化する結果ではない。まず固定 fixture と既存の
Train-heldout の小規模 ablation で loss が raw countdown を修復するか確認し、改善しない
場合は GRU 表現または HUD observability の反例を調べる。

## 保護確認

- Stage A report SHA256:
  `c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`
- Stage A は `passed: false`、理由は
  `gru_near_event_mae_not_20pct_better_than_constant_and_single`
- `validation_used_for_selection: false`; `may_run_stage_c: false`
- Stage A gate を迂回せず、既存 checkpoint を変更しなかった
- ユーザーの未コミット `configs/training/prestige1.toml` 差分は未変更・未commit

## 再実行

CPU:

```powershell
& .venv/Scripts/python.exe scripts/diagnose_timing_countdown_v3.py `
  --checkpoint checkpoints/prestige1/autoplay_sequence_student_v3-e07d934.pt `
  --output reports/timing-countdown-v3-reproduction.json `
  --status-file logs/timing-countdown-v3-reproduction.status.json `
  --device cpu
```

GPU は最後の引数を `--device cuda` に変更する。出力先が存在する場合は上書きを拒否する。

## 成果物

- machine-readable report: `reports/timing-countdown-v3-18de286.json`
- status: `logs/timing-countdown-v3-18de286.status.json`
- stdout: `logs/timing-countdown-v3-18de286.stdout.log`
- stderr: `logs/timing-countdown-v3-18de286.stderr.log`（0 byte）
- evaluator: `src/dmdod/evaluation/timing_countdown_v3.py`
- runner: `scripts/diagnose_timing_countdown_v3.py`
- tests: `tests/test_timing_countdown_v3.py`

## 検証状態

- countdown evaluator と既存 sequence evaluator の関連 test: 12 passed
- 最終診断: exit code 0、stderr 0 byte
- 全 suite の試行: 47 passed 後、今回のコードではなく sandbox の pytest temporary
  directory 作成拒否により setup error で停止
- `git diff --check`: passed
