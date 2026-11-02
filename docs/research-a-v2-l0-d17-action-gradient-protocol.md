# 研究A v2-L0-D17 — 初回PPOの行動別方策勾配を分解（実行前固定）

**事前登録。** D16 Draft PR #59 のHEAD `8fe2bf00fa7b618caaafc015eee32d49213b5fdc` から独立ブランチ/Draft PRを作る。研究Aの基準点はR2 Draft PR #44。リファクタリング監査Draft PR #34とは独立。正式研究002は未承認。本書だけを先にコミットし、実験コード・検査・Actionsを後から追加する。結果を見て条件・停止・判定を変更しない。

## 単一の問い
D16でleft行動だけ平均advantageが正だったことは、初回PPOの**実際の方策パラメータ勾配**にどう現れるか。選択された4行動（none/right/left/both）別に、元のPPO policy objectiveに由来する勾配を分解し、全方策勾配・元のPPO全損失勾配と照合する。行動頻度・左右交代率・指使用率は成功条件にしない。物理打鍵の獲得は測定対象外。

## 凍結入力・条件
- D12成功Actions #41424435292 artifact `research-a-v2-l0-d12-frozen-ppo-cpu` の `frozen-batch.pt` と `capture.json`/`native.json`/`default.json`/`comparison.json` を読み取り専用で再利用。D8成功Actions #41397065687 artifact `research-a-v2-l0-d8-cpu-dispatch` の `native-initial-state.pt` と初期記録を再利用。
- D13/D15で保存済み記録とSHAを検査。A0単指、seed0、8エピソード、128目標、3,800観測、Hit0、Extra9、物理KeyDown9が一致しなければ停止。
- Python 3.11、PyTorch 2.5.1+cpu、CPU native（ATEN_CPU_CAPABILITY未設定）、1スレッド。モデルは元ActorCritic、元PPOのadvantage標準化、Categorical(logits)、ratio、clipped objective、VALUE_COEF、ENTROPY_COEFをそのまま使用。最初のPPO epoch、更新前の重みのみ。新規学習エピソード、PPO更新、Adam step、Motor、物理KeyDownは全て0。勾配の読み取りに `torch.autograd.grad` を用いるがoptimizer.stepはしない。
- 各行動aの勾配は `-sum_{i:action_i=a} objective_i/N` のパラメータ微分（**行動内平均ではなく全N=3800で割る**）。4行動の勾配、全行のpolicy loss勾配、value+entropyの補助勾配、全PPO loss勾配を同じ計算グラフから求める。7回のautograd.gradで、パラメータを更新しない。
- 全パラメータについて各行動勾配のL2ノルム、全PPO勾配との内積・余弦、actor.biasの4要素を保存。全方策勾配と4行動の和、全PPO勾配と（4行動の和＋補助勾配）の最大絶対差を記録。行動別の寄与の符号は「選択確率の増減」や「物理KeyDownの因果効果」と同一視しない。
- D16の行動選択数 (1204/1170/783/643) と初回policy_loss (1.2360120926757645e-08) の照合を行う。全勾配の有限性、4行動が全3,800行を分割すること、モデルの全重みビット列が不変であることを確認する。

## 事前停止・判定
- `valid_action_gradient_decomposition`：入力SHA・物理記録・D16行動数・損失が一致、全勾配が有限、モデルが不変、両再構成誤差が**最大絶対差1e-6以下**であれば、各行動の勾配の大小・符号に関係なく採用する。float32の総和順序による微小差のみ許容する。
- `invalid_gradient_decomposition`：ファイル欠落、SHA/物理記録不一致、非有限値、モデルの変更、D16不一致、誤差1e-6超、構造不一致、計算失敗の場合。失敗ログ・部分成果物を保存し、成功を主張しない。
- 過去のD12初回PPO全勾配と数値が完全一致することは、CPUモデル差を踏まえ必要条件にしない。新しい学習や重み更新をしない。低速Hit率の改善、A1/A3、介入、歴史的D3/D5の再現失敗原因、教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認を主張しない。

## 保全
D16をbaseとした**新規Draft PR**でのみ作業し、既存PR #34/#37〜#59、元の事前登録・重み・成果物を上書きしない。長時間ジョブがあれば重複実行せず停止。**自動マージ・Draft解除・マージは永久禁止。**
