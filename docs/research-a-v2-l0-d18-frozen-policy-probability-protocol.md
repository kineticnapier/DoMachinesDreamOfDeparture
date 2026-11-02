# 研究A v2-L0-D18 — 凍結PPO更新前後の行動確率（実行前固定）

**事前登録。結果・解析コード・検査より先に本書を独立ブランチへコミットする。** D17 Draft PR #60 の HEAD `e845390fe5a03f92c649ae820be2c66202210397` から分岐し、D17をbaseとする新規Draft PRでのみ作業する。研究Aの基準点PR #44、リファクタリング監査PR #34は変更しない。正式研究002は未承認。結果に合わせて条件・停止・評価基準を変更しない。

## 単一の問い
D12で**実際に2 Adam step更新した**A0単指の方策について、同じ3,800個の凍結観測に対する4行動（none/right/left/both）の**出力確率**は更新前後でどう変化したか。D17の勾配ノルムやD16の平均優位度を「確率が上がる」と読み替えず、直接測る。成功判定に左右交代率・指使用率は使わない。

## 固定データ・比較
- D8成功run #41397065687、artifact `research-a-v2-l0-d8-cpu-dispatch` の `native-initial-state.pt`、`native.json`、`default.json`。D12最終成功run #41424435292、artifact `research-a-v2-l0-d12-frozen-ppo-cpu` の `frozen-batch.pt`、`capture.json`、`native.json`、`default.json`、`comparison.json` を**読み取り専用**で再利用。元モデルSHAとバッチSHAはD8/D12の既存検査関数で照合する。初期重み・D12更新後重みは各パラメータのfloat32ビット列で照合。
- A0単指、seed0、8エピソード、128目標、3,800観測、Hit0、Extra9、物理KeyDown9、D16選択行動数 none1204/right1170/left783/both643を要求。1条件でも不一致なら停止。
- 比較条件は `before`（D8元初期重み）、`after_native`（D12 nativeで2 Adam step後の全重み）、`after_default`（D12 defaultで2 Adam step後の全重み）。同一の凍結観測を同じ現在のCPU native経路、torch 2.5.1+cpu、1スレッド、model.eval()、no_grad()で前向き計算。D12のnative/defaultは**更新を行った過去のCPU条件**であり、今回の前向き計算のCPU経路は3条件共通とする。
- 4行動の更新前後の平均確率、各行動の平均差、増加/減少/同値の観測件数、最大絶対差、選択された行動の確率の平均差、argmaxの変更件数、観測ごとの確率分布のL1距離・KL(before||after)の平均と最大を測定。各条件の確率が有限・[0,1]・各行の合計が1（絶対誤差1e-6以下）であることを検査する。3,800行の順序を変更しない。
- 陰性対照として更新前の同じモデルを2回計算し、**確率テンソルのビット単位の完全一致**を要求。保存済み重みの再構成は読み取り専用で行い、元重みとD12保存済みビット列を不変とする。追加学習エピソード0、PPO更新0、Adam step0、Motor0、新規物理KeyDown0。

## 事前停止・判定
- `valid_frozen_policy_probability_comparison`: 入力SHA・全重みビット列・物理記録・行動数・D12の2 Adam step履歴・確率分布・陰性対照がすべて整合し、3条件の測定が有限なら、増減の符号や大きさに関係なく採用する。
- `invalid_frozen_policy_probability_comparison`: 必要ファイル欠落、SHA/形状/ビット列/物理記録不一致、非有限、確率の正規化失敗、陰性対照不一致、再構成不能、実行失敗の場合。失敗ログ・部分JSONを保存し、成功を主張しない。結果を見て許容誤差や条件を変更しない。
- 「確率が増えた」だけで物理KeyDown、Hit、Extra、教師なし運指獲得が改善したとは判定しない。保存済みD12物理記録Hit0/128、Extra9、物理KeyDown9を明示。更新後のMotor再評価は**今回実行しない**。A1二指/A3未学習対照/介入、R1/R2歴史的再現不一致、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。

## 保全・実行
- 別作業ブランチ `research/a-v2-l0-d18-frozen-policy-probability` と独立Draft PRで、事前登録コミットを先に固定してからコード・テスト・Actionsを追加。既存PR #34/#37〜#60、身体、報酬、PPO、元重み、過去成果物は上書きしない。
- GitHub Actionsは専用workflowでPRへのpushを契機に1回実行し、結果JSON、検査・実行ログを成功・失敗にかかわらずartifactへ保存。既存長時間研究ジョブがあれば重複起動しない。
- **自動マージ・Draft解除・マージは永久禁止。**
