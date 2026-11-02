# 研究A v2-L0-D11 — D8実評価の動的観測列を固定したCPU中間演算比較（実行前固定）

**状態：実験コードの作成・実行・結果確認前に固定。正式研究002は未承認。** D10は13固定観測でsoftmaxの2要素差を観測したがEntropyは13入力すべて一致した。D8の45評価試行での平均Entropy差がどの実観測・演算で起きるかを調べる。DMDODリファクタリング監査PR #34の実装監査とは独立する。

## 変更する唯一の因子

同じD8初期重みと**同一の固定された動的観測列**に対し、独立PythonプロセスのATen CPU capabilityだけを変更する。nativeは環境変数ATEN_CPU_CAPABILITY未設定、defaultはプロセス起動前にdefault。CPUモデル、OS、PyTorch 2.5.1+cpu、1スレッド、初期重みSHA、観測float32ビット列は一致が必須。native/defaultのCPU capabilityが異なることも必須。物理モデル・報酬・PPO・重み・観測形式・行動選択式は変更しない。

## 事前固定した入力と採取方法

- 読取専用のD8 Actions #41397065687、artifact `research-a-v2-l0-d8-cpu-dispatch` 内 `native-initial-state.pt`、`native.json`、`default.json`。D8の初期重みSHA一致とdefault側の共通重み読み込みを検証。
- A0 seed0の**学習前**、元D8のargmax 9試行（210/240/250ms × 開始0.835/1.075/1.290s）とsampled 36試行（各4反復、seed=320000+100*case_index+repetition）、計45試行・720目標。追加学習は0試行、PPO更新0回。
- 先にnativeの元`LoggedLowSpeedEnv`を45試行だけ決定論的に再評価し、**各stepの行動選択直前**の8次元観測をfloat32ビット列として順序付きで固定保存する。元の`evaluate_episode`と同じ`torch.softmax`、argmax/`torch.multinomial`、専用乱数生成器を用いる。nativeのHit/Extra/物理KeyDown時刻・左右・判定・controls・step数・各試行の平均EntropyがD8 nativeの保存行と厳密一致しなければ停止する。D8 native/default間の物理列とcontrolsが厳密一致することも要求する。
- 固定観測ファイルを読み取るnativeとdefaultの**独立2プロセス**はMotorを動かさず、元ActorCriticの中間演算（input, shared.linear1, shared.tanh1, shared.linear2, shared.tanh2, actor.logits, critic.value, softmax, entropy）を各観測で評価し、各段階のfloat32ビット列を保存する。D10のforward_stagesをそのまま使用し、元forwardとの一致を確認する。

## 判定と停止条件

1. 45試行の全stepを固定順序で走査し、各段階の異なる要素数、最初に差の出る**段階（段階順を優先）・試行・step・要素・双方のビット値と数値**、最大絶対差を保存する。差がなければゼロ件と報告する。
2. 各試行のEntropyは元`evaluate_episode`と同じPython float累算→step数で除算し、45試行の各modeの平均は元`evaluate_set`と同じ重み付き集約で計算する。両条件の平均EntropyをD8の保存値と**厳密一致**で照合する。許容誤差は事前に設けない。D8平均値が再現できなければ`archived_entropy_not_reproduced`で停止・失敗として記録し、原因特定とは扱わない。
3. 条件成立かつD8の両Entropy再現、かつ差ありなら`dynamic_stage_divergence_observed`、差なしなら`no_dynamic_stage_divergence`。CPUや重み、観測、物理再評価、Entropyの一致条件に失敗した場合は`invalid_diagnostic`または`archived_entropy_not_reproduced`。検査失敗・例外はログとartifactに残し、判定を成功に読み替えない。
4. 既存のHit/Extra/物理KeyDownは**native採取時の再評価のみ**新規に測定し、defaultは固定観測のforwardのみ（Motor step0回）。予想値はD8 native: argmax Hit0/Extra0/KeyDown0、sampled Hit5/Extra44/KeyDown49。これらの不一致は停止。A1/A3の新規評価はしない。
5. 過去のD3/R2・D5/R1の歴史的再現失敗の原因、学習勾配への影響、教師なし運指獲得、人間同等性、単指物理限界、正式研究002の承認は本診断から主張しない。

## 保存・禁止事項

新規Draft PRをD10 PR #53のheadブランチをbaseとして隔離。プロトコルを**実験コードとActionsより先にコミット**し、判定基準を結果から変更しない。新規実験コード、単体テスト、Actions結果JSONと全比較入力・ログを保存。既存PR #34/#37〜#53、学習済み重み、過去の実験結果、Motor・報酬・PPO本体は変更しない。長時間の同一実験が実行中なら二重起動しない。自動マージ、既存PRマージ、Draft解除は永久禁止。
