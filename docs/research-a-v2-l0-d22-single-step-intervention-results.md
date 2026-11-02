# 研究A v2-L0-D22 — 3.620秒の左指1step介入（実測結果）

**判定：`valid_single_step_intervention`、`supports_local_command_contribution=true`。教師なし運指獲得は未達。** D21で観測した29msの初回物理KeyDown時刻差は、A0の3.620秒（step362）の左指指令を1stepだけ反転する介入で**両方向に29msずつ反転**した。ただし、他の身体条件・時刻・seedへの一般化、唯一の原因、固定の身体遅延、人間同等性を証明したものではない。

## 1. 事前登録と実行

- [実装前の事前登録コミット `7c28e173`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/7c28e173db803b4aa0eb82b04f83eb972d08a616) に条件・比較・停止・判定を固定。実装・失敗・結果後も変更していない。
- 独立Draft PR #65（base D21 #64）。監査PR #34、基準点PR #44、過去の研究PRは変更なし。身体・報酬・PPO・重みも変更なし。
- 入力：D8初期重み（Actions #41397065687）、D12 native初回PPO更新後重み（#41424435292）、D19物理結果（#41485667328）、D21全step履歴（#41505006857）。元のSHAとD19/D21原本全stepを照合。
- A0単指、case4/repetition3、240ms間隔、開始1.075s、sampled seed320403、16目標、10ms制御・1ms物理、torch 2.5.1+cpu、1スレッド、native経路。各stepで元の行動抽選を**1回だけ**行い、step362の実行指令だけを置換。以降の方策・身体の反応は自然に進行。
- 実行：B（更新前）、A（更新後）、B→left、A→none、B陰性対照、A陰性対照の**計6 Motor試行**。新規学習0、PPO更新0、Adam更新0。全試行で計**16物理KeyDown**。

## 2. 主要測定値（最初の物理KeyDown）

| 条件 | step362で実行した左指指令 | 初回KeyDown | 目標11との時間差 | 判定 | Hit | Extra | 物理KeyDown |
|---|---|---:|---:|---|---:|---:|---:|
| B 更新前 | none（-1） | **3.718s** | +3ms | Hit | 1 | 1 | 2 |
| A 更新後 | left（+1） | **3.689s** | -26ms | Extra | 0 | 3 | 3 |
| **B→left** | **left（+1）に置換** | **3.689s** | -26ms | Extra | 0 | 3 | 3 |
| **A→none** | **none（-1）に置換** | **3.718s** | +3ms | Hit | 1 | 2 | 3 |
| B陰性対照 | none（-1） | 3.718s | +3ms | Hit | 1 | 1 | 2 |
| A陰性対照 | left（+1） | 3.689s | -26ms | Extra | 0 | 3 | 3 |

- **B→left − B = −29ms**、**A→none − A = +29ms**。事前登録した両方向の時刻条件を満たした。
- B→leftは初回KeyDownとHit/Extra/KeyDown件数がAと一致した。A→noneは初回KeyDownとHitがBに戻ったが、Extraは**2件**でBの1件には戻らなかった。したがって全試行の結果が完全に入れ替わったわけではない。
- 各陰性対照は480step全履歴が完全一致。B/Aの非介入結果はD21全step・D19保存物理traceと完全一致。
- 介入前のstep362の8次元観測はB/Aで一致し、介入試行も各元試行とstep361まで完全一致。実際に置換したのはstep362の**1指令のみ**。その後の身体状態・観測・抽選行動の変化は介入の下流効果として含む。

## 3. 否定的結果と解釈の制限

- **初回PPO更新そのものの改善は確認できない**。非介入のAはBよりHit 1→0、Extra 1→3で悪化。D19全36 sampled対でもHit 5→4、Extra 44→46。
- 介入で初回KeyDown時刻を反転できても、A→noneのExtraはBと一致しない。3.620秒の指令だけで**全ての後続の打鍵を説明できる**という主張は否定的。
- 同一身体・同一seedの局所介入であり、物理KeyDownの29ms差に対する局所的寄与は支持されるが、身体の固定遅延が29ms変化したという証拠ではない。介入後の方策の反応も含むため、他時刻の寄与を排除できない。
- A1二指、A3未学習対照、他seed、長期学習、実機ADOFAI、人間同等性、単指物理的不可能性、正式研究002の承認は未検証・未証明。左右使用率・交代率は成功条件に使用していない。

## 4. 検証・失敗の保全

- 最初の [Actions #41514949759 — FAILURE](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41514949759)：7テストは合格したが、3.620秒の浮動小数点表現に対する厳密等号で `intervention time mismatch`。失敗ログ・結果JSONは[artifact #13175541375](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41514949759/artifacts/13175541375)に保存。比較条件・判定は変更せず、時刻検査だけ `abs(time−3.62)<1e−8` に修正（コミット `ed6a52aa`）。
- [Actions #41514998272 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41514998272)：**7 passed、警告1件**。凍結モデル・原本traceの検証、6 Motor試行、全step・全物理KeyDown記録、結果JSON保存まで成功。
- [artifact #13174512418](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41514998272/artifacts/13174512418)：`result.json`、`steps.json`（6試行の全480stepと物理KeyDown対応）、`tests.log`、`run.log`。30日保存。測定値の要点は本書に永続保存。
- 結果文書を含む最終HEADでの再実行・検証は別途確認する。過去のモデルや成果物は上書きしない。

## 5. 次の問い

3.620秒の局所介入で初回KeyDownが29ms動いた理由を、身体状態の差（位置・速度・押下遷移）に限定して検証することが候補。今の結果は**1試行の局所因果効果**であり、学習失敗全体の原因ではない。新しい因子の検証は別の事前登録・独立Draft PRに固定する。

**Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
