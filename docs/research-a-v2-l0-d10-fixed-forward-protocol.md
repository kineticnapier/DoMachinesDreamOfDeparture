# 研究A v2-L0-D10 — 固定重み・固定観測によるCPU別中間演算診断（実行前固定）

**状態：実装・計測・結果確認前に固定。正式研究002は未承認。** D3〜D9の再現失敗を受けた診断であり、盲検の新規運指獲得実験ではない。

## 問いと変更する唯一の因子

D8では同じ初期重みでもATen CPU capabilityの切替で学習前の平均Entropyに差が生じた。一方、元R1/D3のEntropyはどちらの条件でも再現できず、D3/R2とD5/R1の再現失敗の原因は未確定である。**固定した同一の初期重み・同一の観測を使い、CPU演算経路（native と ATEN_CPU_CAPABILITY=default）のみを変えたとき、ネットワークのどの段階で最初の厳密な数値差が出るか**を調べる。

条件は同一GitHub Actionsジョブ内の独立した2 Pythonプロセス。nativeはATEN_CPU_CAPABILITYを設定しない。defaultは起動前にdefaultを設定する。実際のtorch.backends.cpu.get_cpu_capability()が異なることを確認する。両条件のCPUモデル、PyTorch版、スレッド数、OSと入力・重みのSHAが異なれば無効。

## 固定資料・入力

- D8の成功run #41397065687、artifact #13119278298の native-initial-state.pt / native.json / default.json を読み取り専用で使用。元R1の初期重みSHAが両D8記録で一致することを確認する。重みは両条件で同じファイルをロードし、再初期化だけを比較しない。
- Python3.11、PyTorch2.5.1+cpu、CPU単一スレッド、PYTHONHASHSEED=0、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。
- 観測は元のLoggedLowSpeedEnvを**resetするだけ**で得る9ケース（interval_ms=210/240/250、start_s=.835/1.075/1.290の直積）と、事前固定した次の4個の8次元人工入力：全0、全1、[-1,-.5,0,.125,.25,.5,.75,1]、[1,-1,.75,-.75,.5,-.5,.25,-.25]。全入力をfloat32で固定し、入力のビット列SHAを両条件で一致させる。人工入力は物理的に到達可能とは主張しない。
- 元のActorCritic（shared: Linear(8,32)→Tanh→Linear(32,32)→Tanh、actor: Linear(32,4)、critic: Linear(32,1)）を変更しない。元のevaluate_episodeと同じsoftmaxとEntropy式を用いる。

## 計測・判定

1. 各13入力について、input、shared.linear1、shared.tanh1、shared.linear2、shared.tanh2、actor.logits、critic.value、softmax、entropy のfloat32ビット列を保存。元ActorCritic.forwardと分解計算のlogits/valueが各条件で厳密一致しない場合は停止。
2. 同じ段階・同じ入力のビット列を要素単位で比較し、**最初の差がある段階・入力番号・要素番号・両数値・最大絶対差**を報告。差がない場合も全件一致を保存する。
3. stage_divergence_observed（実機CPU・重み・入力一致かつ異なるdispatchで差あり）、no_stage_divergence_on_fixed_probes（同条件で全段階一致）、dispatch_not_distinct、invalid_diagnosticのいずれかで判定。差の有無を問わず、**過去のD3/D5の原因特定や学習中の勾配再現と混同しない**。
4. 事前基準に許容誤差を設けず、後から入力・判定基準を追加しない。テストや保存資料の検証に失敗したら失敗として残す。

## 停止・保存・変更禁止

学習0試行、PPO更新0回、Motor step 0回、新規物理KeyDown0回。既存R1/R2/A0/A1/A3のHit・Extra記録は参照のみで再採点しない。CPU実験が既に実行中なら重複起動しない。結果JSON・テスト・ログ・失敗理由を新規Actions artifactへ保存し、結果文書へ追記する。元のPR #34/#37〜#52、モデル・身体・報酬・PPO・学習済み重み・過去結果は変更しない。新規Draft PRのみ作成し、**自動マージ・Draft解除を永久に禁止**。
