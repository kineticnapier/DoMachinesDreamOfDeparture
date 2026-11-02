# 研究A v2-L0-D24 — 1ms物理再生の実測結果

**判定：`valid_1ms_physical_replay`（SUCCESS）。教師なし低速打鍵の学習改善は確認されない。**

## 事前登録・作業の分離
- [実行前に固定した条件](research-a-v2-l0-d24-1ms-physical-replay-protocol.md) はコミット [`0f82dee`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/0f82dee0736dfececa3d53218dd501c953b15c4f)。解析コード・テスト・結果の作成前に単独で保存し、変更していない。
- 独立Draft PR [#67](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/67)、baseはD23 Draft PR #66のHEAD `4faa223d`。基準点 #44、監査 #34、既存研究PR、Motor本体、報酬、PPO、モデル重みは変更なし。
- D22の最終成功[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995)／[artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296)の6系列×480制御stepの実行指令を固定し、通常の `Simulation.step` で各制御stepを10回の1ms物理tickに分割して再生。方策の再推論・再学習・新しい介入はなし。
- [最初の成功Actions #41532764194](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41532764194)：**9テスト合格**、実測再生成功。全結果JSON、3.620〜3.730秒の1ms軌跡、テスト/実行ログは[artifact #13182248226](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41532764194/artifacts/13182248226)（保持30日）に保存。重要な数値は本書に永続記録する。

## 1msの押下閾値横断
押下閾値は2.000mm。各行の「直前→直後」は実際に実行した連続する1ms物理tickであり、10ms状態の補間ではない。

| 系列 | 直前時刻 | 直前左位置 | KeyDown時刻 | KeyDown時左位置 | 物理判定（D22原本） |
|---|---:|---:|---:|---:|---|
| 更新前B | 3.717s | 1.967875mm（未押下） | **3.718s** | **2.022194mm（押下）** | Hit |
| 更新後A | 3.688s | 1.985879mm（未押下） | **3.689s** | **2.039232mm（押下）** | Extra |
| B→left | 3.688s | 1.985879mm（未押下） | **3.689s** | **2.039232mm（押下）** | Extra |
| A→none | 3.717s | 1.967875mm（未押下） | **3.718s** | **2.022194mm（押下）** | Hit |

- 更新前Bの3.717s直前の閾値までの距離は**0.032125mm**。3.718sで閾値を**0.022194mm**超えた。
- 更新後Aの3.688s直前の閾値までの距離は**0.014121mm**。3.689sで閾値を**0.039232mm**超えた。
- 左指指令の単一step介入で最初のKeyDown時刻はB→leftが**29ms早まり**、A→noneが**29ms遅くなる**ことを1ms再生で確認。陰性対照2系列は対応する元系列と全1ms履歴が完全一致。
- D22の全480stepの10ms開始状態（位置・速度・押下・活性・疲労・協調）と全KeyDown時刻・指・件数を独立した通常物理経路で照合。最大許容誤差は状態 `1e-9`、時刻 `1e-8` 秒。全件合格。
- **総計28,800物理tick、再生した物理KeyDown16件、6系列。** Hit/Extraは新たに採点しておらず、D22の保存済み判定を転記した。

## 否定的結果と未解決事項
- D22保存済みの局所比較では、B（Hit1、Extra1、KeyDown2）、A（Hit0、Extra3、KeyDown3）、B→left（Hit0、Extra3、KeyDown3）、A→none（Hit1、**Extra2**、KeyDown3）。1stepを戻して初回Hitが回復しても、余分な打鍵は元のBまで回復しない。
- D19の非介入sampled評価は576目標で**Hit5→4、Extra44→46、物理KeyDown49→50**。今回の1ms再生は学習改善を示さず、この否定的結果を変更しない。
- **身体状態のみによる媒介因果は未証明。** D22の実行指令列には介入後の方策反応が含まれており、D24は保存済み指令を固定して通常の物理経路で再生したもの。追加の身体状態固定介入や他seedへの一般化は行っていない。
- A1二指、A3未学習対照、長期学習、人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。左右交代率・使用率を成功条件にしていない。
- 実行時のテスト失敗はなし。ActionsのNode.js非推奨警告は出たが検査・再生・artifact保存は成功。

**自動マージ・Draft解除・既存PRのマージは行わない。**
