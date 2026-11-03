# 研究A v2-L0-D33 — step413の初回PPO勾配方向と左指確率感度（事前登録）

**探索的・未承認研究。実装、検査、解析、結果取得より前に本文だけを独立コミットする。結果を見て条件・基準・停止条件を書き換えない。**

## 単一の問い
D27で物理KeyDown 4.327sへの因果効果を確認したA0 seed320403のstep413において、D17の**最初の凍結PPOバッチの初回勾配**（行動none/right/left/bothの方策損失とvalue+entropy補助損失）が、更新前の同一観測での左指指令確率 `p=P(left)+P(both)` を**微小な勾配降下**で増やす方向か減らす方向かを測定する。これはAdam 2 step後の実測確率変化を因果分解する実験ではない。

## 固定原本・条件
- base: D32 Draft PR #75 の `research/a-v2-l0-d32-step413-parameter-group-swap`、新規head: `research/a-v2-l0-d33-step413-frozen-gradient-projection`。新規Draft PRのみ変更。基準点 #44、監査 #34、過去の研究PRは変更しない。
- D8 Actions #41397065687 の初期重み `native-initial-state.pt` と検証用JSON、D12 Actions #41424435292 の凍結バッチ `frozen-batch.pt` と native 初回PPOの保存記録、D17 Actions #41472753103 の `result.json`、D28 Actions #41562899406 の `result.json` を読取専用で使用。D12の初期SHA・バッチSHA、D17の行動別勾配ノルム/選択数とD28のstep413観測・4行動確率を照合する。D32は参照のみ。
- Python 3.11、torch 2.5.1+cpu、native CPU、1スレッド、同一初期重み。凍結8試行/3,800観測/128目標、選択数1204/1170/783/643、Hit0/Extra9/物理KeyDown9を確認。
- D17と同じ最初のepochの正規化adv、元PPOのclip、`policy_loss`、`value_loss`、`entropy`、`VALUE_COEF` と `ENTROPY_COEF`。行動別損失は全3,800行で割る。勾配は4行動・policy合計・補助・PPO合計を `torch.autograd.grad` で得る（7回）。追加で固定step413の `p` の勾配を1回求める（計8回）。`optimizer.step`・学習・Motor・採点は禁止。
- 全1509パラメータを順序固定で扱い、各損失 `L_i` の感度 `s_i=-∑_θ (∂p/∂θ)(∂L_i/∂θ)` をfloat64内積で算出（単位SGD学習率あたりの一次近似）。行動4群・補助・policy合計・PPO合計、shared/actor/critic群別も保存する。各群・各行動の加算再構成誤差は1e-7以下。ゼロや負の値も有効な結果とする。
- 初期重みと固定バッチのSHAがD12と一致し、D17の選択数完全一致、行動別勾配L2ノルムの絶対差1e-5以下、D28更新前の4確率と再推論値の最大差5e-6以下、D17の再構成最大誤差1e-6以下、元モデルの全パラメータbit不変、有限値、critic-onlyの`∂p/∂critic`が0、各寄与の再構成誤差1e-7以下を**成立条件**とする。いずれか不成立は `invalid_step413_gradient_projection` として失敗を保存。全成立なら符号に関係なく `valid_step413_gradient_projection`。

## 停止条件・解釈
- 上記4件の原本取得と単一観測の1回の感度解析、専用単体テストのみ。追加seed・PPO更新・Adam更新・再学習・Motor・物理KeyDown・Hit/Extra採点は禁止。長時間ジョブが存在すれば重複起動しない。
- `s_i`は**仮想的な微小SGD更新**の方向微分。Adamの正規化、2回目のepoch、非線形性、実際の重み更新後の確率を説明する値ではない。行動別の原因・報酬への因果帰属を主張しない。D28の実際の差は +0.001216411590576、D19非介入576目標のHit5→4、Extra44→46、物理KeyDown49→50は**改善なし**として固定して併記。
- A1/A3、他seed、長期学習、教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002承認は未証明。
- 成功・失敗ともJSON/テスト/実行ログをActions artifactへ保存し、結果文書とDraft PRへ否定的結果も記録する。**自動マージ・Draft解除・既存PRマージは永久禁止。**
