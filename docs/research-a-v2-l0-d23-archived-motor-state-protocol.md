# 研究A v2-L0-D23 — D22保存済みMotor状態の閾値到達過程（実行前固定）

**予備診断の事前登録。正式研究002の承認ではない。** 本文を解析コード・テスト・結果より先に単独コミットし、測定後に条件・比較・停止・判定を変更しない。

## 研究対象と分離
- 独立ブランチ `research/a-v2-l0-d23-archived-motor-threshold-trace`、base D22 Draft PR #65の確定HEAD `ac3d0f88c4d345dd8874df799ec08a15524c1b88`。独立Draft PRを作る。基準点 #44、監査 #34、過去研究PR、Motor本体・報酬・PPO・重み・過去の結果は変更しない。
- **単一因子：D22のstep362（3.620s）で左指の実行指令をnone↔leftに変えたときの、保存済み身体状態の推移**。介入を追加しない。D22で行ったB→leftとA→noneの結果を読み取り専用で比較する。新規学習0、PPO/Adam更新0、Motor評価0、新規物理KeyDown0。
- 入力は成功した最終HEADのActions #41515190995、artifact #13175252296 `research-a-v2-l0-d22-single-step-intervention` 内の `result.json` と `steps.json` のみ。GitHub Actionsの `download-artifact` で取得し、D22元成果物を変更しない。

## 事前固定の比較
- 全6系列：B（更新前）、A（更新後）、B→left、A→none、B陰性対照、A陰性対照。A0単指、case4/repetition3、240ms、開始1.075s、sampled seed320403、10ms制御・1ms物理、480制御step。16目標。実KeyDownの初回はB=3.718s Hit、A=3.689s Extra、B→left=3.689s Extra、A→none=3.718s Hit。B/AのHit・Extra・物理KeyDownはそれぞれ1/1/2、0/3/3。介入系列は0/3/3、1/2/3。
- D22の結果分類 `valid_single_step_intervention`、介入回数、6系列の全step数、陰性対照の全履歴完全一致、step362より前の対応系列の完全一致、step362での介入内容と事前観測一致、全物理KeyDown件数・時刻・分類とD22結果JSONとの整合を検査する。不一致・欠落・非有限値・時刻不連続は即停止し `invalid_archived_motor_state_trace` としてログ・失敗JSONを保存。
- 観測するのは `pre_motor` の左位置[m]、左速度[m/s]、左押下状態と、`pre_diagnostics` の左活性、左疲労、手疲労、手協調の**既存の記録だけ**。Motor内部の未保存1ms状態を補間・創作しない。押下閾値は凍結Motorの `KeyConfig.actuation_m=0.0020m`、離鍵閾値は `reset_m=0.0018m`。この2値は比較用の定数で、物理モデルの変更ではない。
- 固定時点：step362=3.620s、363=3.630s、365=3.650s、368=3.680s、369=3.690s、371=3.710s、372=3.720s。各系列の位置、速度、活性、閾値までの距離（2mm−位置）、押下状態、選択/実行指令を保存する。B対B→left、A対A→noneの各状態項目の**最初の差のstep**と値を報告する。差がない項目はnullのまま保存する。
- 各系列の初回物理KeyDownの直前制御stepと、その直後に記録された制御stepの状態を報告する。10msの状態と1msのKeyDown時刻を混同しない。各介入対で初回KeyDown時刻差[ms]、Hit/Extra/KeyDown件数差を併記する。

## 判定・停止・否定的結果
- 全照合が成立すれば `valid_archived_motor_state_trace`。位置・速度・活性が期待した向きに変わらなくても成功とし、否定的な測定結果をそのまま保存する。
- 補助的な `early_trace_closer_to_threshold_at_step368` は、B→leftのstep368における押下閾値までの距離がBより**小さい**ときだけtrue。同値・逆方向はfalse。A→noneのstep368における距離がAより**大きい**ときだけ `reverse_trace_farther_from_threshold_at_step368` をtrueとする。両方trueでも**身体状態を媒介した因果関係の証明ではない**。介入後の方策反応を含み、未保存の1ms内部状態は再構成できない。
- 学習成功はD19の独立基準（sampled Hit増加かつExtra非増加）のまま。D22/D23の介入でHitが戻っても学習成功とは扱わない。D19全体のHit5→4、Extra44→46、物理KeyDown49→50は変更しない。
- A1二指・A3未学習対照・他seed・長期学習・人間同等性・単指の物理的不可能性・正式研究002の承認は未証明。左右使用率・交代率は成功条件ではない。

専用検査、結果JSON、実行/失敗ログをGitHub Actionsで `if: always()` 保存し、結果の要点と失敗を独立Draft PR/結果文書に永続記録する。**自動マージ・Draft解除・既存PRのマージは永久禁止。**
