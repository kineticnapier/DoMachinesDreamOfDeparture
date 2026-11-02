# 研究A v2-L0-D11 — D8の実評価観測21,110件でsoftmaxとEntropyの差を特定

**判定：`dynamic_stage_divergence_observed`。GitHub Actions SUCCESS、専用テスト9件合格、警告1件。正式研究002は未承認。**

## 1. 実行前固定と保存先

- [実験前固定プロトコル](research-a-v2-l0-d11-dynamic-forward-protocol.md)：実装前コミット `42ff405f4879a1ad3b2e14f1046421f2c60783ea`。
- [Draft PR #54](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/54)：D10のDraft PR #53をbaseに隔離。元PR #34と#37〜#53には変更なし。
- [GitHub Actions #41419638175](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41419638175)：**SUCCESS**、9 passed / 1 warning。実験コード・テスト・Actionsの実行コミット `308a51eb4668905b93cc857fc2e9bd7a385cc215`。
- [全観測・全中間演算・比較結果・ログ artifact #13129520795](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41419638175/artifacts/13129520795)。圧縮済み30,712,352 bytes。比較JSONには段階別件数、最大差、最初の差、D8との厳密照合結果を含む。
- 読み取り専用のD8成功run #41397065687、artifact #13119278298の初期重み・native/default評価記録を使用。

## 2. 実際の評価観測を固定した比較

D8のA0 seed0、学習前45試行（argmax 9、sampled 36）、計720目標をnativeで一度だけ再評価した。全21,110 stepの**行動選択直前**の8次元float32観測を保存。nativeの全45行のHit/Extra、物理KeyDown時刻・左右・判定、行動数、step数、平均EntropyはD8 nativeの保存行と厳密一致した。D8 native/defaultの物理列・controlsも一致を確認。

この同じ観測列を、独立したnative/defaultの2プロセスで元ActorCriticの中間演算へ入力した。D11実機のCPU命令経路はnative **AVX512**、default **NO AVX**。同一CPUモデル・OS・PyTorch・1スレッド・初期重みSHA・観測SHAを確認した。D10のnative AVX2と実機命令経路が異なることに注意する。

| 段階 | 異なるfloat32要素数（全21,110観測） | 最大絶対差 |
|---|---:|---:|
| 入力 | 0 | 0 |
| Linear 1 / Tanh 1 / Linear 2 / Tanh 2 | 0 | 0 |
| Actor logits / Critic value | 0 | 0 |
| **softmax** | **5,686** | **8.940696716308594×10⁻⁸** |
| **Entropy** | **228** | **2.384185791015625×10⁻⁷** |

softmaxの異なる要素はargmax 1,109、sampled 4,577。Entropyの異なる観測はargmax 35、sampled 193。softmaxの差が生じた観測は合計3,923件（argmax 767、sampled 3,156）。

**最初の差はsoftmax**：argmaxの最初の試行、step 0、2番目の確率。native `0.24430598318576813`、default `0.24430596828460693`、差 `1.4901161193847656e-08`。**最初のEntropy差**は同じ試行のstep 64で、native `1.3582208156585693`、default `1.3582210540771484`（差 `2.384185791015625e-07`）。

## 3. D8平均Entropyを厳密に再現

| 評価 | native | default | default − native |
|---|---:|---:|---:|
| argmax（9試行） | 1.3561848983669778 | 1.3561848978587430 | −5.082347875884352×10⁻¹⁰ |
| sampled（36試行） | 1.3558157407760280 | 1.3558157402183817 | −5.576463735×10⁻¹⁰（概数） |

両条件・両modeの平均Entropyは、D8の保存値と**ビット単位相当のPython float厳密一致**。したがって、D10の13固定入力では再現できなかったEntropy差が、D8の実際の動的観測列の中に存在し、softmax以降に生じることを確認した。

物理再評価の測定はargmax Hit 0/Extra 0/KeyDown 0、sampled Hit 5/Extra 44/KeyDown 49。新規学習0試行、PPO更新0回。nativeでMotorを21,110 step再評価したが、固定観測のnative/default forward比較ではMotor step0回。A1/A3の新規評価はしていない。

## 4. 否定的結果と未解決事項

- 21,110観測でsoftmax 5,686要素、Entropy 228観測に差があっても、D8のnative/defaultでは**行動数・物理KeyDown・Hit/Extraは一致**した。今回の数値差が打鍵成績を悪化させた証拠はない。
- この実験のnativeはAVX512で、D10のnativeはAVX2。**CPU命令経路が同じでも実機条件が違えば結果を安易に一般化できない**。D8の保存済み平均Entropyを厳密に再現した事実は、過去R1/D3/D5のCPU情報欠落を補うものではない。
- D3のR2再現失敗、D5のR1再現失敗、学習中の勾配・PPO更新への影響は**未証明**。元R1/D3の初期Entropy（1.3561848871858129）も今回の2条件では再現できない。
- 低速打鍵の教師なし運指獲得は依然未達。単指物理限界・人間同等性・正式研究002承認も未証明。左右交代率や使用率を成功条件にはしていない。

**次の切り分け候補**：保存済みの初期評価観測に対して、softmaxのどのCPU演算が1 ULP差を生むかを分解する。ただし実際のPPO学習失敗への因果的説明には別の検証が必要。新しい実験は条件・比較対象・停止条件・評価基準を再び実行前に固定する。

既存の研究・リファクタリング用PR、身体、報酬、PPO、重み、過去の結果は未変更。**PR #54はDraft・未マージ。自動マージ・Draft解除は禁止。**
