# 研究A v2-L0-D8 — CPU演算経路の変更で学習前Entropyが変化（ただしD3/D5原因は未確定）

**判定：`dispatch_effect_observed`。D8診断の条件比較は成立し、計測上の差を確認。教師なし運指獲得は未達、正式研究002は未承認。**

## 1. 実行前固定と記録

- [実行前固定プロトコル](research-a-v2-l0-d8-cpu-dispatch-protocol.md)：実験コードより前のコミット `e2c22250dfbcb724bba9d2af48f95c68519bd100`。
- [Draft PR #51](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/51)：D7 Draft PR #50をbaseにした隔離ブランチ。
- [最終Actions #41397065687](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41397065687)：**SUCCESS**、単体テスト11件成功、警告1件（NumPy未導入に関するPyTorch警告）。元R1/D3/D7を照合し、学習なしでnative/defaultの2条件を実行。
- [最終artifact #13119278298](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41397065687/artifacts/13119278298)：検証JSON、各条件の全45評価試行・物理KeyDown・CPU設定、比較JSON、実行ログ、共有初期重み。ZIP SHA-256 `73598e9af721bcbc1e16625b4defafef27f50365ab9cabe99184c5d1503aa1be`。
- 元R1 Actions #41309837474、D3 Actions #41369778494、D7 Actions #41391082538。元の結果とチェックポイントは読み取り専用。
- **新規学習0試行・PPO更新0回**。A0 seed0のargmax9試行＋sampled36試行を各条件で評価しただけ。身体・報酬・PPO・観測・学習条件は未変更。

## 2. CPU命令経路の変更で初期評価Entropyに差

同じActionsジョブ、同じCPUモデル・PyTorch 2.5.1+cpu、CPU単一スレッドで、ATenの命令経路だけを変更した。

| 条件 | 実際のCPU命令経路 | 初期argmax平均Entropy |
|---|---|---:|
| 元R1/D3（過去記録） | 過去の実行経路は本診断では未特定 | 1.3561848871858129 |
| D7（過去記録） | 過去の実行経路は本診断では未特定 | 1.3561848983669778 |
| D8 native | AVX2 | **1.3561848983669778** |
| D8 default | NO AVX | **1.3561848978587430** |

`default − native = −5.082347875884352×10⁻¹⁰`。最初の厳密不一致は `stage0.argmax.mean_entropy`。

**同一初期重み・同一CPUで命令経路を変更すると、学習前Entropyの厳密な値が変化する**ことを確認した。D8 nativeはD7の既知値と一致したが、元R1/D3の値にはどちらも一致しなかった。今回の差は元R1/D3とD7の差（約1.118×10⁻⁸）を完全には説明しない。

### 重要：同じseedだけでは初期重みも一致しなかった

最初のCIでは、native/defaultでそれぞれ `torch.manual_seed(0)` から初期化すると、初期重みSHA-256が異なり、事前の一致条件を満たせず停止した。

- nativeのseed初期化SHA-256：`5420d95edaeea137bd7e6f0d5fc84c3890c41236f43e4cc4c44bd0392142f83c`
- defaultのseed初期化SHA-256：`4441ba2b171b522e90612db3056365bf57439b2b8c6d11423bbcd6743d55f68b`

この交絡を避けるため、nativeで生成した**元R1と一致する初期重みを、default側へそのまま読み込む**よう診断コードを修正した。最終比較では両条件の初期重みSHA-256が一致している。元の重み・モデル・学習コードは一切変更していない。

同じseedで異なる初期重みになった原因もCPU命令経路の影響と整合するが、どの初期化演算で差が出たかは未特定。初回の停止は失敗記録として保持した。

## 3. Hit/Extraと物理KeyDownは不変

| 評価 | native Hit | default Hit | Extra（両条件） | 物理KeyDown（両条件） |
|---|---:|---:|---:|---:|
| argmax 9試行・144目標 | 0 | 0 | 0 | 0 |
| sampled 36試行・576目標 | 5 | 5 | 44 | 49（左49・右0） |

**全45試行の行動選択数と物理KeyDown列は両条件で厳密一致**。Hit/Extra、実際の左右打鍵にも差はなかった。sampledの平均Entropyには小差があったが、これだけでは学習中のHit/Extra悪化を示せない。

A1二指やA3未学習対照の再評価は行っていない。既存のA0/A1/A3の評価記録も変更していない。

## 4. 途中失敗と修正の履歴

- [初回Actions #41396657998](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41396657998)：FAILURE。単体テスト9/10成功、D7を読む検査の引数不足を修正。
- [2回目の代表Actions #41396748810](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41396748810)：FAILURE。テスト10/10成功、元R1/D3/D7照合とnative評価成功。defaultで初期重みSHA不一致のため停止。後に共通初期重みの読み込みを追加。
- [Actions #41396898976](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41396898976)：SUCCESS。Entropy差を確認したが、比較JSONの`physical_trace_exact`がEntropyを含む全行比較になっていたため、物理列の表示だけを修正。
- **最終Actions #41397065687：SUCCESS**。物理KeyDown列とEntropyを分離して照合し、`physical_trace_exact`と`action_controls_exact`はいずれもargmax/sampled両方でtrue。

途中の失敗や以前のartifactは削除・上書きしていない。判定基準・許容誤差・元の事前登録は変更していない。

## 5. 否定的結果・未解決

- **D8の命令経路差だけでは、D3/D5の歴史的な再現不一致を説明できない。** 元R1/D3の初期Entropyを両条件とも再現できなかった。
- 初期Entropyに差があっても、今回の評価では**物理KeyDown・Hit/Extra・行動選択に差がなかった**。
- 元R1/D3/D5/D7の実行環境でどのCPU命令経路・演算器が選ばれていたかは、今回の2条件比較からは確定できない。
- D3のR2勾配は引き続き元R2の学習時の勾配として採用できない。最初の異なるPPO更新も未特定。
- 人間同等性、単指の物理的不可能性、教師なし運指獲得、正式研究002の承認は未証明。

**次回の優先課題**：既存のR1/D3/D5/D7の実行環境ログを読み、CPUモデル・命令経路・初期重みを比較する。差が不足していれば、初期評価を固定観測・固定重みの演算単位に分解する独立計画を立てる。元の研究PR #34/#37〜#50は変更しない。自動マージ・Draft解除は禁止。
