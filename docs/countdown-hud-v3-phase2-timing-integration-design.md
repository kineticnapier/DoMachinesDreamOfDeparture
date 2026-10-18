# HUD v3 Phase 2 — Countdown Timing Integration Design

## Scope and decision

Phase 2ではHUD v3のCountdown観測をTimingへ接続するが、既存Timing v1.4の
263D入力、アルゴリズム、出力を変更しない。新しいcomposite層が278D frameを
受け取り、先頭263Dを凍結v1.4へ、末尾15Dと観測metadataを新規Countdown phase
trackerへ渡す。Fingering、Motor、checkpoint、Validation20には接続しない。

Countdown表示は周期の位相anchorであり、初回target時刻そのものではない。
特に`Go!`から初回targetまでの固定delayは禁止する。

### BPM observation contract

現行263DのTile BPM / Real BPMは、JRP等のModが画面へ数値表示する場合に限る
`mod-visible-tile-real-bpm-v2`観測契約である。stock ADOFAIのCountdown表示だけから
数値BPMを直接観測できる、という意味ではない。CompositeとContact HUD adapterは
この契約を明示的に要求し、`stock-only-no-numeric-bpm`入力をfail closedする。
stock-only経路でchart設定や内部BPMを観測値に代入してはならない。凍結Timing v1.4の
263D仕様とアルゴリズム自体は変更しない。

## Proposed interfaces

新規ファイルは次の3単位に分ける。

1. `planning/countdown_phase_tracker_v01.py`
2. `planning/countdown_first_event_v01.py`
3. `planning/hud_timing_composite_v20.py`

policy-facing contract案:

```text
CountdownPhaseUpdate
  step
  observed_at_s
  status: ready | numbered | go | hidden | invalid | stale
  displayed_digit: int | None
  transition_observed_at_s: float | None
  measured_tick_period_s: float | None
  tick_period_uncertainty_s: float | None
  remaining_transitions_to_go: int | None
  evidence_age_s

TimingCandidate
  local_id
  due_local_s
  uncertainty_s
  first_seen_local_s
  source: countdown_first_event | timing_v14
  confirmations
  confirmed_now

TimingFrameUpdateV20
  step
  status
  pending
  new_ids, reused_ids, canceled_ids
  abstention_reason
```

これらに`countdownTicks`、target/floor ordinal、chart time、AutoPlay時刻、
judge結果を含めない。`displayed_digit`は実画面から認識された値なので許可する。

## Causal state estimation

`CountdownPhaseTracker`は現在・過去の有効な表示だけを使用する。

- `Ready!`だけでは総Countdown回数が不明なので初回時刻を出さない。
- 最初の数字で残り表示遷移数は観測できるが、未知の
  `countdownSpeedMultiplier`を1と仮定しない。
- 連続する2つ以上の数字遷移からtick周期を測る。10 ms samplingと各frameの
  capture latencyから周期の上下限を保持し、単一点の真値にしない。
- 数字が1ずつ減らない、frameが欠損・古い、遷移intervalが整合しない場合は
  `invalid_sequence`として棄権する。
- `Go!`は独立した位相anchorとして使う。消去後のhiddenを次のeventと解釈しない。
- invalid frameでは状態を推測更新せず、evidence ageと不確実性だけを増やす。
- 新しい有効cueが過去の予測区間と矛盾したら、未確定candidateをcancelする。

10 ms gridでは真の境界直後の最初の観測まで0超10 ms以下の量子化遅延がある。
frame timestamp/capture latencyも加え、transition区間を
`(previous_rendered_at_s, current_rendered_at_s]`として保持する。

## First-event estimator

`CountdownFirstEventEstimator`はphase updateと、同じ時刻の可視HUD263 geometry/BPM
だけを受け取る。初回targetの候補は次をすべて満たす場合だけ出す。

1. 数字列またはGoでCountdown位相が一意。
2. tick周期に有限の観測区間がある。
3. 現在床のentry/exit角、惑星位置、回転方向が可視履歴で一意。
4. floor-0の残り回転時間が現在/過去の角速度から有限区間として求まる。
5. due区間の下限が現在より未来で、候補同士が44 ms Lead判定を曖昧にしない。

計算は概念的に次の区間和とする。

```text
first_due_interval = observed phase anchor interval
                   + remaining displayed tick intervals
                   + visible floor-0 remaining-arc interval
```

表示BPMは整合性checkに使えるが、観測済みtick間隔や回転速度を隠れた設定値で
置換しない。Pause、checkpoint、fast takeoff、Midspin、MultiPlanet、方向alias、
未知markerでは初期実装は棄権する。SetSpeedは初回targetまでに画面上で数値が
確定しない場合、既存v1.4と同じく境界を越えて外挿しない。

## Composite routing

- Countdown中の初回eventだけを新規estimatorの責務とする。
- 通常playに入った後は凍結Timing v1.4の結果をそのまま採用する。
- 両者が同時に候補を出した場合、単一のv1.4候補だけがdue区間と重なるときに限り
  同一local IDへ統合する。重ならない余剰候補や複数候補は、正解時刻で選別せず
  `source_conflict`としてすべて棄権する。
- 矛盾時に正解時刻で選ばず`source_conflict`として棄権する。
- Countdown中にcueが欠損・stale・invalidになったframeでは、raw v1.4 forecastは
  診断用に保持するがplayable候補へ公開しない。
- v1.4へ渡すのは常に`frame.legacy_hud263`のみで、278Dを誤入力しない。
- Countdown trackerの内部状態をv1.4へ埋め込まず、両者を個別reset可能にする。

local IDの対応付けは予測区間の重なりだけで行う。評価器のtarget IDは使わない。
同一候補のframe反復を新規eventとして数えない。
前frameからpendingに残るlocal IDは、そのframeで未確定でも`reused_ids`へ含める。
`new_ids`と`reused_ids`はpendingを分割し、消えたIDは`canceled`または`elapsed`の
どちらか一方に入る。新しいCountdown due区間が既存区間と重ならない場合は、同じ
local IDを更新せず`countdown_interval_conflict`で既存IDをcancelする。

## Uncertainty, confidence, and abstention

Phase 2ではcalibrated confidenceを作らない。代わりに根拠を明示する。

- cue sequence validation
- transition confirmation count
- tick-period interval
- due-time interval
- evidence age
- abstention reason

44 ms判定は`due_lower_bound - observed_at_s >= 0.044`のときだけ
`lead_guaranteed`とする。中心推定だけが44 msを超えていても成功扱いにしない。
capture latencyが不明なら実画面評価では保証不能として棄権する。

## Leakage boundary

renderer、policy、evaluatorを別objectにする。

```text
renderer: internal game state -> actually rendered frame
sensor: rendered frame -> HUD v3 observation
policy: current/past HUD v3 -> Timing candidates
evaluator: completed predictions + AutoPlay truth -> posthoc metrics
```

policy-facing型とログをAST/schema検査し、次を禁止する。

```text
countdownTicks, GetCountdownTime, first_tick_s, tick_period_s(renderer truth),
target_time_s, target_index, target_ordinal, floor_index, chart_time,
segment.targets, _used, _missed
```

テスト用rendererの内部scheduleは画面生成にのみ使い、policy呼び出し引数に渡さない。

## Minimum implementation and test order

1. `CountdownPhaseTracker`を純粋な観測列fixtureで実装。
2. 厳密境界、10 ms量子化、欠損、stale、digit skip、pitch、可変tick周期を試験。
3. first-event estimatorを一定BPM・通常開始の合成geometryだけで実装。
4. N=3/N=5、120/240/480 BPMでdue区間と44 ms Leadを評価。
5. composite adapterで263D prefixがv1.4へbit-for-bit同一に渡ることを確認。
6. 同一HUD列で「v1.4単独」と「v1.4+Countdown」を比較。
7. 通常playのv1.4出力が追加sensorの有無で不変であることを回帰試験。

ここまでをTrain由来または合成fixtureだけで行う。学習、checkpoint更新、
Validation20、Qualification/Reserve/FINAL、Fingering/Motor閉ループは実施しない。

## Predeclared Phase 2 gates

Phase 2実装の成功条件:

- N=3/N=5の初回candidateが評価器truthに対し10 ms以内、かつ重複なし。
- 120/240/480 BPMの通常開始fixtureでfalse candidate 0。
- 44 ms Leadは中心値でなくdue下限で判定し、確保不能例は正しく棄権。
- 欠損/stale/矛盾digit列で未来値を補完せず棄権。
- 通常playにおけるTiming v1.4のstatus、local ID、due時刻が完全一致。
- policy-facing schemaとcapture関数に禁止truth field参照がない。

失敗時は閾値探索へ進まず、量子化、cue検出、period推定、geometry推定、
reconciliationのどこで失敗したかを分類する。
