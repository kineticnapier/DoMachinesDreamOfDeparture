# HUD v3 Phase 10 — Countdown / v1.4 Source Identity Reconciliation

## 判定

`SOURCE_IDENTITY_RESOLVED`

Triple Cross初回イベントの`source_conflict`は、未知の2イベントを時刻の近さだけで
統合した問題ではなかった。Countdownとv1.4が既に一意に照合済みだったにもかかわらず、
同じv1.4 local IDの微小なpoint due変動を毎frame再照合したため、照合が失効していた。

観測可能な初回一意照合と、その後のv1.4 local ID continuityだけを使ってaliasを維持する
限定修正を実装した。target truth、chart event ordinal、AutoPlay時刻はID判定に使用していない。

## 原因

対象は`02521 - Triple Cross - Milkyway.adofai`の初回イベントである。

| frame | episode time | legacy状態 | lifecycle |
|---:|---:|---|---|
| 91 | 0.91 s | `countdown:0`生成 | new ID 0 |
| 112 | 1.12 s | `v14:17`初出。一意に重なり、ID 0へ結合 | reused ID 0 |
| 113 | 1.13 s | 同じ`v14:17`を再照合 | reused ID 0 |
| 114 | 1.14 s | `source_conflict` | canceled ID 0 |
| 115 | 1.15 s | `v14:17`を別イベントとして再公開 | new ID 1 |
| 120 | 1.20 s | v1.4側ID 1をelapsed | elapsed ID 1 |

step 112の結合時は、Countdown区間に重なるv1.4候補が`v14:17`の1個だけで、競合する
別候補はなかった。結合後の公開区間は
`[1.199999998456672, 1.199999998456672] s`だった。

step 114では、Countdown自身の未変更の区間更新後の点が
`1.199999998538823 s`、同じ`v14:17`のpoint dueが
`1.199999997495537 s`となった。差`1.043286 ns`が既存の重なりEPS `1 ns`を越え、
単一候補照合が失敗した。v1.4 IDは変化しておらず、追加候補も存在しなかった。

このため、原因はTiming推定失敗ではなく、確立済みsource identityを毎frameのpoint
interval overlapだけで再証明しようとしたLifecycle/Compositeの問題である。

結果JSONには初回イベント前後の各frameについて、HUD278全278要素、Countdown cue、
phase、estimate区間、v1.4候補、Composite pending/new/reused/canceled/elapsedを保存した。

## 因果的な同一性規則

aliasは次の全条件を満たす最初のframeだけで作る。

1. Countdown first-event tokenが既に存在する。
2. そのdue区間と重なるv1.4候補が正確に1個だけ存在する。
3. 重ならない追加v1.4候補が存在しない。
4. 以後の継続は、v1.4 trackerが`reused`として維持する同じlocal IDに限定する。

alias確立後は後続v1.4 pointの微小変動でCountdown dueをretimeせず、Countdown側の
既存推定・区間更新を維持する。別のv1.4候補が同時に現れた場合はidentityを推測せず、
従来どおり`source_conflict`とする。

したがって「時刻が近い候補一般」を結合する規則ではない。v1.4 IDが消失・変更した場合、
または初回照合に競合がある場合も結合しない。不確実性を追加規則で縮小していない。

## Train bool密集イベント回帰

Train boolの最初の2イベント間隔は約20.829866 msである。

| event | truth | reconciled ID | first lead | 結果 |
|---:|---:|---:|---:|---|
| 0 | 3.999333624 s | 0 | 79.333624 ms | 従来と同じCountdown予測 |
| 1 | 4.020163490 s | 1 | 20.163490 ms | 独立したv1.4イベント |

両イベントのmatching ID集合は互いに素で、隣接イベントの誤統合はなかった。初回の
due interval `[-9.500239, +0.499761] ms`（truth相対）も変化していない。2打目はTiming
監査上は別IDとして観測されるが、Phase 9で確認した44 ms Motor Lead不足は解消しない。

## 固定Train回帰監査

Phase 7固定4譜面（12秒）とTrain bool冒頭5秒、合計200イベントを、Lifecycle v1.5の
ままidentity reconciliationだけOFF/ONして比較した。

| 指標 | legacy conflict | reconciled |
|---|---:|---:|
| Clean Lifecycle | 198 / 200 | 199 / 200 |
| Event ID不整合 | 1 | 0 |
| 正当な棄権 | 1 | 1 |
| False Positive ID | 0 | 0 |

譜面別ではTriple CrossだけがClean `63→64`、category 5 `1→0`となった。他の4譜面は
全categoryが不変で、既存成功イベントの回帰は0だった。Triple CrossのMidspin直後
event 23はcategory 3の正当な棄権を維持した。

Triple CrossのLifecycle総数は次のとおり。

| arm | new | reused | canceled | elapsed |
|---|---:|---:|---:|---:|
| legacy conflict | 65 | 435 | 1 | 64 |
| reconciled | 64 | 437 | 0 | 64 |

重複newと不要なcancelが各1件なくなり、初回イベントはID 0のままstep 120でelapsedした。
Adapterは全frameを受理し、ID契約違反は0だった。

due差分とuncertainty差分は、reconciliation対象外のID・譜面では全て0。対象aliasでは
step 113にdue差分が1frameだけあり、差は約0.443 nsだった。これは新しいTiming推定や
v1.4 retimeではなく、確立済みaliasに対して後続v1.4 pointを再交差しなかった結果である。
alias uncertainty差分は0で、不確実性を縮小していない。

## 契約と範囲

全事前宣言GateがPASSした。

- Triple Cross初回conflict解消
- 重複ID生成解消
- False Positive ID増加なし
- reconciliation対象外のdue / uncertainty差分なし
- 既存成功イベントの回帰0
- Train bool初回Countdown回帰なし
- Train bool密集2イベントの誤統合なし
- Adapter全frame PASS
- 合成Motor受信契約PASS
- Midspin棄権維持

Phase 9の物理結果commit `8fbc818b84ad11c439f27b10f306b55c3218ce25`は変更せず、
実Motor A/Bは制約どおり再実行していない。Fingering、Motor、Scorer、checkpoint、
Countdown推定式、v1.4 due推定式は変更していない。

## 再現

```powershell
.\.venv\Scripts\python.exe -u `
  scripts/audit_hud_timing_phase10_source_identity_v01.py `
  --output reports/hud-timing-phase10-source-identity-<new-id>.json

.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_audit_hud_timing_phase10_source_identity.py `
  tests/test_hud_timing_composite_v20.py `
  tests/test_hud_v3_composite_forecast_source_v30.py `
  tests/test_hud_timing_phase8_lifecycle_integration.py `
  tests/test_persistent_fingering_scheduler_v01.py `
  tests/test_timing_fingering_motor_integration_v02.py
```

採用結果は`reports/hud-timing-phase10-source-identity-20261019-r3.json`。
Validation20、FINAL、学習、checkpoint更新、実Motor評価は実施していない。
