# DMDOD Phase 11 — Unseen Train16 Failure Diagnosis

## 判定

**AUDIT_COMPLETED / EVENT_DIAGNOSIS_COMPLETE / NO_CODE_CHANGE**

アップロードされたユーザー実行済みの全量JSON
`hud-timing-phase11-generalization-local-r1.json`（SHA256:
`e9e7788797671146571bc734ece294d096478aef86d6e232c1419366bbc79350`）
を読み取り、全16件の未成功イベントを個別に追跡した。
同時にユーザー提供の`Train.zip`に含まれる譜面を既存`compile_adofai` /
`build_playable_segment`で事後解析し、実際のfloor index、Midspin、
Twirl、SetSpeedと突き合わせた。これは**推論後の診断**であり、
HUD推論にchart truthを渡していない。

| 対象 | C: v1.5 + reconciliation |
|---|---:|
| 未評価Train | 16 charts、1,068 target events |
| Clean Lifecycle | 1,052/1,068 = **98.5019%** |
| Category 2 / 3 / 4 / 5 | **0 / 14 / 2 / 0** |
| 完全Cleanな譜面 | **11/16** |
| False-positive ID | **0** |
| 既存固定5譜面 | 199/200 |
| 全structural_checks | PASS |

## 未成功16件の所在

| Chart | Clean/Targets | 未成功idx（category） | 事後floor構造 |
|---|---:|---|---|
| ALL MY FELLAS | 88/92 | 64,69,74,78（3） | 全4件がMidspin直後、前打鍵から12.5ms |
| ADOFAI Regular Dan - 2nd Dan | 117/118 | 114（3） | Midspin直後、前打鍵から529.4ms |
| Crossing Notes | 34/35 | 23（3） | Midspin直後、前打鍵から187.5ms |
| First town of this Journey remix | 211/220 | 1,36,41,46,146,151,156（3）、174,201（4） | 7件がMidspin直後、2件がTwirl直後 |
| ヒアソビ | 54/55 | 54（3） | 当該floorにSetSpeedとCheckpoint |
| その他11譜面 | 全イベントClean | なし | 事後に失敗を観測せず |

Midspin直後の打鍵13件では、直前floorがMidspin（0°、非打鍵）で、
そのMidspinと次の打鍵floorは**同じ到達時刻**である。
これは正規の`PlayableChartTarget`であり、
Midspin floorだけが非打鍵扱いになっている。
共通するHUD側の失敗状態は`special_future_floor`。
ただしMidspin後の打鍵が早いケースばかりではない
（Regular Danは前打鍵から529.4ms、Crossing Notesは187.5ms）。
**高速密集だけを原因とは判断できない。**

First townのcategory 4はevent 174（t≈9.889706s）と
201（t≈10.948529s）。いずれも1360BPM、直前floorに`Twirl`、
直前打鍵との間隔は約**7.352941ms**。前後のevent分類はCleanで、
当該eventには一致するforecast IDが存在しない。
その80ms前窓のstatusには`eligible_partial`しかなく、
`_evaluate`は`observability_unverified`（category 4）と分類した。
**これは観測不能性の証明ではない**。HUD上の旋回方向履歴、
短間隔イベントの二重表現、10ms制御フレーム境界を
切り分ける診断が必要。

ヒアソビのevent 54（t≈11.641791s）は
当該floorに`SetSpeed`と`Checkpoint`があり
`special_future_floor`による棄権。
この1件をMidspin問題に混同しない。

## v1.4からの13件の救済

13件すべてが`03810 - Parallel Universe Shifter - Eraser`。
event index:
`30,31,32,33,34,35,37,38,39,41,42,43,44`。

- v1.4: 全13件がcategory 5（ID/lifecycle inconsistency）
- v1.5: 全13件がcategory 1
- 同譜面のcanceled: **15→1**
- 同譜面のelapsed: **32→46**
- **成功した共通1,039件の初回forecast MAEはA/B/Cすべて0.020120684ms**。
  合計MAEの`0.020121→0.019872ms`変化は
  clean母集団の変化であり、Timing estimator自体の改善ではない。

## Source Identity ON/OFFの注意

B（v1.5、reconciliation OFF）とC（ON）は
イベント分類・false positive・Lifecycle集計が同じ。
**ただし全frameの予測値が同じという意味ではない。**
8譜面に各7frame、合計56frameのCountdown関連due差分がある
（同数のCountdown区間差分、uncertainty差分なし）。
例：Rainbow Chaserのstep 243–249で約0.14–1.67nsの
Countdown due差がある。これはsource identityの再交差処理の
微小な数値差であり、今回の分類を変えていない。

## 推定器の実装・再生時系列と照合した追加診断

`hud_timing_tracker_transition_v14.py`の`_direction`では、次floorの
可視フラグ`Midspin (slot[6])`や`Checkpoint (slot[12])`が立つ場合に
`special_future_floor`で棄権する。よって上記13件のMidspin直後の欠測は
**実装された保守的な打切り規則と対応**している。
ヒアソビはSetSpeedとCheckpointが同じfloorにあり、
この分岐で直接打切り条件になるのは**Checkpoint (slot[12])**。
SetSpeed (slot[9])単独は同じ規則の直接原因ではない。
したがって、ヒアソビを単に「SetSpeedの未来数値が未知」とみなさない。

First townのcategory 4をHUD-only sourceで12秒だけ診断再生すると:

| target | 前のTwirlイベント | 事前frame | 次のframe |
|---|---:|---|---|
| 174, due 9.889706 | 9.882353 | step 988 (9.88s) : `eligible_partial`、pendingは前イベント9.882353のみ | step 989 (9.89s) : 9.889706は既に過去、次イベント9.926471のみ |
| 201, due 10.948529 | 10.941177 | step 1094 (10.94s) : `eligible_partial`、pendingは前イベント10.941177のみ | step 1095 (10.95s) : 10.948529は既に過去、次イベント10.985294のみ |

つまり、**可視Twirlを含む将来floorの先を予測しない保守的規則**と
**10ms制御刻みに対して短い7.353ms間隔**の組合せによって、
次のframeを待つ間に対象が通過している。
根本的な観測不能性の証明ではなく、
「観測済みのTwirlフラグから事前にイベント時刻を導出できるか」
という因果的な未解決問題である。

## 今後の優先順位

1. **Midspin→次の打鍵の因果的な到達時刻・Event ID導出**
   （13件）。非打鍵Midspinと同時刻に発生する次floorを、
   利用可能な可視情報だけで分離できるか調べる。
   観測情報不足が証明されない限り`unobservable`とはしない。
2. **Twirl直後の7.353ms間隔の打鍵2件**。
   未検証扱いを放置せず、推定器の見逃しなのか、
   frame interval境界・方向履歴・二重発火判断の問題なのかを
   HUD-only反事実評価で調べる。
3. **SetSpeed / Checkpointがあるヒアソビの1件**。
   Mod可視HUDからSetSpeedの数値到達情報を事前に得られるか
   別の観測可能性問題として扱う。

以上は事後の構造相関からの**調査優先順位**であり、
解決可能性の実証や新モデルの成功率予測ではない。

## 実行・変更範囲

既存の固定Train16結果JSONを**再実行・改変しない**。
このPhase 11解析ではTiming estimator、Fingering、Motor、Scorer、
checkpoint、学習、Validation20、FINALに変更なし。
詳しい機械可読リストは
`reports/hud-timing-phase11-event-diagnosis-20261019-r1.json`。
