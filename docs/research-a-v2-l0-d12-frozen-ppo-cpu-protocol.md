# 研究A v2-L0-D12 — 固定した最初のPPOバッチでCPU演算経路だけを変える（実行前固定）

**事前登録。実験コード・Actions実行・結果確認より前にコミットする。** 正式研究002は未承認。D11は固定観測21,110件でsoftmax 5,686要素・Entropy 228件の数値差を確認したが、実際のPPO更新への影響は未検証。リファクタリング監査PR #34の作業とは独立する。

## 唯一の変更因子
PyTorch 2.5.1+cpuのATen CPU capability。独立した同一GitHub Actions runner上のPythonプロセスで、native（環境変数未設定）とdefault（`ATEN_CPU_CAPABILITY=default`）を比較する。CPUモデル、OS、Python、PyTorch、スレッド数1、初期モデル重み、全PPOバッチのfloat32ビット列、元の行動、旧対数確率、報酬・return・value、Adam初期状態、PPO係数と実装は完全に同一。CPU capabilityが実際に異なることが必須。身体、報酬、観測形式、PPO本体を変更しない。

## 固定入力・採取方法
- D8の成功run #41397065687から初期重み`native-initial-state.pt`と`native.json`/`default.json`を読み取り専用で再利用し、D8の初期重みSHAを検査する。
- A0単指・seed0、元L0の`train_episode`をそのまま使い、Python `random.Random(0)`で**最初の8エピソードだけ**採取する。学習前の重みで実際のMotorを動かし、全観測、行動、old_log_probs、old_values、returns、scoreと物理KeyDownの時刻・左右・Hit/Extraを固定して保存する。追加の学習エピソードは8、実際のMotorによる新規物理KeyDownは採取分のみ。
- 採取したバッチと元重みを両CPU条件に読み込み、`update_ppo`を**元のまま1回呼ぶ**。これは2 epoch/2 Adam step、学習バッチの新規採取や追加エピソードはしない。保存対象は更新前の同一性検査、元PPO返却loss、更新後全パラメータのfloat32ビット列、Adam内部状態、実行環境。
- 事前に固定した比較は、(1)更新後パラメータで異なる要素数と最大絶対差、(2)Adam exp_avg/exp_avg_sqの異なる要素数、(3)PPO返却loss各値の差。初回差のパラメータ名・要素番号・両ビット値を記録。ゼロ差も否定的結果として保存。
- 採取時の8エピソードの目標128件についてHit・Extra・物理KeyDownと全イベントを保存する。A1/A3の新規評価はしない。A0/A1/A3の過去比較を変更しない。

## 事前停止・評価基準
- 8エピソード以外の学習を禁止。両条件で同じ初期重みSHA、同じバッチSHA、CPUモデル/OS/PyTorch/スレッド数、異なるCPU capability、finiteな損失/重みを確認できない場合`invalid_diagnostic`として停止し、ログ・部分成果を保存する。CPUの異なる命令経路を再現できなければ`dispatch_not_distinct`として停止。
- すべての条件が成立し、PPO更新後の全重みとAdam状態が一致なら`no_ppo_update_difference_observed`、差があれば`ppo_update_difference_observed`。どちらも研究Aの学習失敗の原因確定ではない。許容誤差を結果から導入しない。
- 既存のD3/R1の履歴的な更新の完全再現はこの実験の必要条件ではなく、**歴史的PPO勾配**として扱わない。初期評価と元R1の不一致を解消したとは主張しない。D3/R2・D5/R1の再現失敗との因果関係、Hit改善、人間同等性、単指物理限界、正式研究002の承認を主張しない。
- 同一実験が進行中なら重複実行しない。出力ファイルの上書き禁止。失敗時もActions artifactとログを保存する。

## 作業境界
D11 PR #54のheadをbaseとした独立ブランチ/Draft PRでのみ作業する。**本事前登録コミットを作成してから**実験コード・テスト・Actionsを追加する。既存PR #34/#37〜#54、身体・報酬・PPO・元重み・元の結果を上書きしない。自動マージ、Draft解除、既存PRのマージは永久禁止。
