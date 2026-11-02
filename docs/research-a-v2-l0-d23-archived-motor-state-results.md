# 研究A v2-L0-D23 — 保存済みMotor状態と2mm押下閾値の実測結果

**判定：`valid_archived_motor_state_trace`（SUCCESS）。低速打鍵の学習改善なし。** D22の6系列を読み取り専用で解析し、3.620秒の左指指令介入後、位置・速度・活性の最初の差が**3.630秒**に発生することを確認した。3.689秒に早い側だけが2mm押下閾値を超えて押下済みとなり、29ms早い物理KeyDownに対応した。これは保存済み10ms間隔の状態との整合であり、1ms内部軌跡の再構成や媒介因果の証明ではない。

## 1. 事前登録・入力・実行

- 実装より前の事前登録コミット：[`17751e00`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/17751e00e2135995faa1b404bb2255eda7e57501)、[固定プロトコル](research-a-v2-l0-d23-archived-motor-state-protocol.md)。事後変更なし。
- 独立Draft PR #66、base D22 Draft PR #65のHEAD `ac3d0f88`。基準点 #44、監査 #34、過去研究PRは変更なし。
- 元データ：D22最終成功[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995)、[artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296) の `result.json` と `steps.json`。全6系列×480制御step、計16個の**過去の**物理KeyDownを照合。A0単指、240ms間隔、開始1.075s、seed320403、16目標、10ms制御・1ms物理。
- 新規学習0、PPO/Adam更新0、Motor評価0、**新規物理KeyDown0**。既存身体・報酬・PPO・モデル重みは不変。
- [成功Actions #41524306993](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41524306993)：専用**10テスト合格**、保存済み結果の解析も成功。全測定JSON、検査ログ、実行ログは[artifact #13178297962](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41524306993/artifacts/13178297962)に保存（30日保持）。要点は本書に永続保存。

## 2. 介入前から最初の状態差まで

| 3.620秒の実行指令 | 更新前B：none | 更新後A：left |
|---|---:|---:|
| step362直前の左位置 | −0.600299mm | −0.600299mm |
| 左速度 | 0.027040m/s | 0.027040m/s |
| 左活性 | −0.01312527 | −0.01312527 |
| 左押下状態 | false | false |

この時点では直前状態は同一。B→leftとA→noneの各介入も、それぞれ元系列とstep361まで全stepが一致する。

**step363（3.630秒）に初めて差が現れた。** BとB→left（後者はAと同じ値）の比較：

| 指標 | B（none） | B→left |
|---|---:|---:|
| 左位置 | −0.388973mm | −0.323670mm |
| 左速度 | 0.01402734m/s | 0.02979269m/s |
| 左活性 | −0.04780971 | +0.02248176 |
| 左疲労 | 0.18145717 | 0.18245692 |
| 手協調 | 0.99683998 | 0.99722754 |

逆介入A→noneではこれらの値が逆方向に入れ替わる。押下状態の最初の差はstep369（3.690秒）。**手疲労は全480stepで介入対の値が完全一致**し、最初の差は存在しなかった（null）。これは「手疲労の変化がこの29ms差の必須条件」という説明には否定的である。

## 3. 閾値直前の測定

押下閾値は2.000mm、離鍵閾値は1.800mm。各行は制御step開始時点の保存済み値であり、step中の1ms内部状態を示さない。

| 時刻 | B / A→none 左位置 | A / B→left 左位置 | B側閾値まで | A側閾値まで | 押下状態 |
|---|---:|---:|---:|---:|---|
| 3.680秒 | −0.225684mm | +1.590173mm | 2.225684mm | 0.409827mm | 両方false |
| **3.690秒** | **+0.170618mm** | **+2.093421mm** | **1.829382mm** | **−0.093421mm** | **B false / A true** |
| 3.710秒 | +1.519522mm | +3.269775mm | 0.480478mm | −1.269775mm | B false / A true |

3.680秒では早い側が押下閾値に**約1.816mm近い**。3.689秒に早い側の実KeyDownが発生し、次の3.690秒の保存状態は押下済みで2mm超。一方、遅い側は3.690秒でも2mm未満で未押下、3.718秒に初回KeyDownとなる。

事前登録した補助判定は **`early_trace_closer_to_threshold_at_step368=true`、`reverse_trace_farther_from_threshold_at_step368=true`**。ただし、閾値到達の途中経過を1ms刻みで記録したわけではなく、身体状態だけを固定する独立介入もしていない。

## 4. Hit / Extra / 物理KeyDownと否定的結果

| 条件 | 初回KeyDown | Hit | Extra | 物理KeyDown |
|---|---:|---:|---:|---:|
| B 更新前 | 3.718s（Hit） | 1 | 1 | 2 |
| A 更新後 | 3.689s（Extra） | 0 | 3 | 3 |
| B→left | 3.689s（Extra） | 0 | 3 | 3 |
| A→none | 3.718s（Hit） | 1 | **2** | 3 |
| B陰性対照 | 3.718s（Hit） | 1 | 1 | 2 |
| A陰性対照 | 3.689s（Extra） | 0 | 3 | 3 |

**学習改善は確認されない。** D19全36 sampled対の非介入評価は576目標に対してHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。今回の解析はこの結果を変更しない。A→noneで初回Hitが回復してもExtraはBの1件でなく2件であり、step362だけで全後続打鍵を説明できない。

## 5. 失敗履歴と限界

- 最初の[Actions #41524223424 — FAILURE](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41524223424) は**9テスト成功・1テスト失敗**。時刻異常を検査する合成テストでB本体のstep370を変更したため、意図した時刻検査より先に「BとB陰性対照が一致しない」検査が失敗した。実験結果の失敗ではなく検査用データの隔離不備。失敗テストログは[artifact #13179416235](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41524223424/artifacts/13179416235)に保存。
- [修正コミット `27f2b021`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/27f2b0211b7cd2b3128b01e02a7ae796c8ee8ca1) で、時刻異常を独立のB→left系列に与えるようテストだけ修正。事前登録・判定・実験コードは変更なし。その後のActions #41524306993はSUCCESS。
- 身体の位置・速度・活性の変化は**局所介入後の記述的な経路**。行動方策の下流反応を含み、1ms内部状態・状態媒介の唯一因果・他seedへの一般化を証明しない。手疲労に差がないことは、この1試行の手疲労変化を原因とする説明に否定的だが、疲労一般の影響を否定しない。
- A1二指、A3未学習対照、他seed、長期学習、人間同等性、単指物理的不可能性、正式研究002承認は未検証・未証明。左右使用率・交代率は成功基準に使わない。

**Draft維持。自動マージ・Draft解除・既存PRのマージは永久禁止。**
