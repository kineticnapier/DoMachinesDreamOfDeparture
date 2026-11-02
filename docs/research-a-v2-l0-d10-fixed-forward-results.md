# 研究A v2-L0-D10 — 固定入力のCPU別中間演算でsoftmax差を確認

**判定：stage_divergence_observed。専用テスト11件合格、GitHub Actions SUCCESS。正式研究002は未承認。**

## 実行前固定・保存先

- [事前固定プロトコル](research-a-v2-l0-d10-fixed-forward-protocol.md)：実装前コミット `4c22597bb99006874f0bfe2c6b2ff53a406c6d44`。
- [Draft PR #53](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/53)：D9 Draft PR #52をbaseに隔離。
- [GitHub Actions #41409627904](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41409627904)：**SUCCESS**、専用テスト **11 passed / 1 warning**（NumPy未導入に関するPyTorch警告）。すべての評価・比較ステップが成功。
- [全13入力・各段階のビット列・比較結果・検証ログ artifact #13125036503](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41409627904/artifacts/13125036503)。成果物はGitHub Actionsに保存。
- 元D8成功run #41397065687 / artifact #13119278298から同じ初期重みを両条件に読み込み、元重み・学習済みモデルは変更していない。

## 1. 中間演算の最初の差

同一GitHub Actionsジョブの独立2プロセスで、ATen CPU capabilityのみを変更した。nativeは**AVX2**、defaultは**NO AVX**。13個の固定float32入力・同一初期重みを使い、元ActorCritic.forwardと分解した中間演算の厳密一致を各条件で確認した。

| 段階 | 異なるfloat32要素数（13入力の合計） |
|---|---:|
| 入力 | 0 |
| 第1 Linear | 0 |
| 第1 Tanh | 0 |
| 第2 Linear | 0 |
| 第2 Tanh | 0 |
| Actor logits | 0 |
| Critic value | 0 |
| **softmax** | **2** |
| Entropy | **0** |

**最初の厳密不一致はsoftmax**。最初の観測（210ms・開始0.835s）の2番目の確率要素で、native `0.24430598318576813`、default `0.24430596828460693`、差は **1.4901161193847656×10⁻⁸**（float32の1 ULP）。元の入力・各Linear・Tanh・Actor logits・Critic valueは13入力すべてでビット単位一致。

これは、少なくともこの固定入力集合では**softmaxの演算経路が確率の最終ビットに影響した**という直接観測である。D8の平均Entropy差をそのまま説明したとは言えない。

## 2. 否定的結果・限定事項

- **Entropyは13入力すべてで厳密一致**。softmaxの確率要素に差があっても、今回の各入力のEntropyには差が現れなかった。D8の45試行全体の平均Entropy差を今回の固定13入力だけで再現したわけではない。
- **学習0試行・PPO更新0回・Motor step0回・新規物理KeyDown0回**。Hit/ExtraやA0/A1/A3の学習成績を新しく測定していない。元D8の物理打鍵結果は変更しない。
- 同じ初期重みで異なるCPU命令経路により数値差が起こることは確認したが、元R1/D3/D5の歴史的なCPUモデル・演算経路は記録がなく、**D3のR2再現失敗やD5のR1再現失敗がCPU由来と証明されたわけではない**。
- softmaxの差はモデル重みやPPO勾配を変更した結果ではない。学習中の行動列やHit/Extraへの影響は未検証。
- 人間同等性・単指の物理的不可能性・教師なし運指獲得・正式研究002の承認は引き続き未証明。

## 3. 次の切り分け

次は必要に応じて、D8の保存済み評価と同じ**実際の動的観測列**を、学習や物理実行を再開せず固定して中間演算を比較する。固定観測の取得方法・停止条件を新たなプロトコルに記載してから行い、D10の事前条件を後付け変更しない。

元PR #34/#37〜#52、身体、報酬、PPO、重み、過去結果は未変更。**PR #53はDraft維持、マージ・Draft解除は禁止。**
