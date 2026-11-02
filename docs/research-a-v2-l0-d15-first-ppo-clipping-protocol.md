# 研究A v2-L0-D15 — 初回PPO逆伝播の勾配ノルム制限の作用（実行前固定）

**本書は解析コード・Actions実行より前に別コミットで固定する。** 研究A v2-L0-R2 Draft PR #44が基準点。D14 Draft PR #57のHEADをbaseに独立Draft PRを作成する。監査PR #34とは別の研究。正式研究002は未承認。

## 単一因子・問い
元PPOの `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` が最初のAdam.step直前に実際に勾配を縮小したかを、D12の**同一凍結バッチ・初期重み**で確認する。D13/D14が記録したCPU別の勾配差は制限**後**であり、制限前の大きさは未測定。今回の比較は制限の前後のみ。学習失敗や数値差の原因を断定しない。

## 入力と再現条件
- D12最終成功Actions #41424435292のartifact #13130809065の `capture.json` / `native.json` / `default.json` / `comparison.json` / `frozen-batch.pt` を読み取り専用で取得。D13の既存検査で記録のSHA、CPU、勾配差292/407、128目標・Hit0・Extra9・物理KeyDown9を照合する。
- D8成功Actions #41397065687の `native-initial-state.pt` / `native.json` / `default.json` を読み取り専用で取得。D12既存の `weights_digest` と `batch_digest` で初期重みと全バッチの同一性を検証する。
- 同一runnerでPyTorch 2.5.1+cpu、Python 3.11、単一スレッド、同じ初期重みとバッチ。native（ATEN_CPU_CAPABILITY未設定）とdefault（ATEN_CPU_CAPABILITY=default）を**別Pythonプロセス**で実行。CPUモデル・OS・PyTorchは一致、CPU capabilityは異なること。
- 元 `a002_exploratory_ppo.update_ppo` の最初のepochの損失計算（観測結合、advantagesの標準化、Categorical、ratio、clamp、policy/value/entropy係数）を変更せず再構成し、`loss.backward()` を**1回だけ**実行。勾配をfloat32生ビットで保存し、元と同じ `clip_grad_norm_(..., 1.0)` を呼び、戻り値の制限前総ノルムと制限後勾配を保存。**Adam.stepは一度も実行しない。** モデル・PPO本体・身体・報酬・観測を変更しない。
- 制限後の全1509要素が、D12保存済み**最初のAdam.step直前の勾配**と両条件で完全一致することを必須とする。一致しなければ `invalid_replay` として停止。D13の292要素差も照合する。

## 事前に固定する測定・判定
- 両条件の制限前総ノルム、制限しきい値1.0、制限係数 `min(1, 1/(norm+1e-6))`、制限前後で変化した勾配要素数と最大絶対差、CPU間の制限前/後勾配差数と最大絶対差、制限後のD12全ビット一致をJSONに記録。非有限値を禁止。
- `norm>1` かつ前後で勾配が変われば `clipping_active`、`norm<=1` かつ前後完全一致なら `clipping_inactive`。両CPUともactiveなら `clipping_active_in_both`、両方inactiveなら `clipping_inactive_in_both`、片方ずつなら `clipping_mixed`。規則に合わない、再現不一致、入力欠落、CPU条件が区別不能なら `invalid_replay`。結果を見て基準を変えない。
- 結果がactiveでも「勾配制限が学習失敗を引き起こした」とは言わない。inactiveなら**この最初のバッチ**では勾配制限が原因ではないと限定する。CPU差の初出演算は未検証。
- 新規学習エピソード0、PPO更新0、Adam更新0、Motor実行0、新規物理KeyDown0。A0単指seed0の過去8エピソード（128目標）だけを使用。A1二指/A3未学習対照の追加実行なし。左右交代率・指使用率は成功条件にしない。

## 停止・保存・作業境界
入力不整合、SHA不一致、CPU差がない、制限後の勾配がD12と一致しない、非有限値、想定外のモデル構造なら停止し、部分ログ・失敗記録をActions artifactに保存。失敗を隠さず再現基準を後から変更しない。結果と実行ログはGitHubに残す。
元PR #34/#37〜#57、過去の事前登録・成果物、身体、報酬、PPO、学習済み重みを変更しない。Draft解除・自動マージ・手動マージを禁止。人間同等性、単指物理的不可能性、教師なし運指獲得、正式研究002承認は未証明。
