# 研究A v2-L0-D9 — 過去5実行の環境証拠を照合（CPU実機は未特定）

**判定：`historical_cpu_unresolved`。監査は成功したが、D3/D5の数値不一致原因は特定できない。正式研究002は未承認。**

## 実行前固定と実行結果

- [実装前固定プロトコル](research-a-v2-l0-d9-historical-runtime-protocol.md)：コミット `40b682d2c9b1ccdbf3514071617b791db38d4649`。
- [Draft PR #52](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/52)：D8 PR #51をbaseとする別ブランチ。Draft維持・未マージ。
- [Actions #41403399640](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41403399640)：**SUCCESS**。専用テスト8件合格。過去の5つのジョブログとD7/D8の保存済みJSONを照合した。
- [全ジョブログ・抽出結果・テスト記録 artifact #13122476660](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41403399640/artifacts/13122476660)：`audit.json`、元5ジョブログ、`tests.log`、`audit.log`。取得ZIP SHA-256 `545dc464ece3a04009c2e7ef0600ed6c0a17c61eca72fe6f414c8ce194dc3516`。
- **新規学習0試行、PPO更新0回、新規物理KeyDown0回**。既存の身体・報酬・PPO・重み・評価記録は変更していない。

## 過去の5ジョブで確認できたこと

| 実行 | ジョブ | runner image | Python | PyTorch | 実機CPUの記録 |
|---|---|---|---|---|---|
| [元R1](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41309837474/job/124857703147) | 124857703147 | ubuntu-26.04 | 3.11.17 | 2.5.1+cpu | **不明** |
| [D3](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41369778494/job/125037538287) | 125037538287 | ubuntu-26.04 | 3.11.17 | 2.5.1+cpu | **不明** |
| [D5](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41380105087/job/125068823089) | 125068823089 | ubuntu-26.04 | 3.11.17 | 2.5.1+cpu | **不明** |
| [D7](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41391082538/job/125102397739) | 125102397739 | ubuntu-26.04 | 3.11.17 | 2.5.1+cpu | AMD EPYC 7763（成果物から） |
| [D8](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41397065687/job/125120568955) | 125120568955 | ubuntu-26.04 | 3.11.17 | 2.5.1+cpu | AMD EPYC 7763（成果物から） |

5ジョブともrunner version `2.338.2`、runner image release `ubuntu26/20261025.179`、`PYTHONHASHSEED=0`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` が一致した。これらはジョブログからの抽出値であり、実機CPUが同じことを意味しない。

D7とD8の保存済み診断JSONには、`AMD EPYC 7763 64-Core Processor`、PyTorch `2.5.1+cpu`、Torchスレッド1、MKLDNN有効と記録されている。**R1・D3・D5のジョブログにはCPUモデルとATen CPU命令経路の明示的記録がない。** D7のATen経路も記録されていない。D8だけは同一実行でnative=`AVX2`、default=`NO AVX`を記録している。

## 学習前の既知のEntropy差（過去記録の再照合）

| 条件 | 初期argmax平均Entropy |
|---|---:|
| 元R1 / D3 | 1.3561848871858129 |
| D7 / D8 native | 1.3561848983669778 |
| D8 default | 1.3561848978587430 |

D7 − 元R1 = `1.1181164882856365e-08`、D8 default − native = `-5.082347875884352e-10`。今回、これらの過去値をD7/D8の成果物と照合したが、学習や評価を新たに行っていない。

D8の保存済みA0評価はargmax 144目標でHit0・Extra0、sampled 576目標でHit5・Extra44・物理KeyDown49。D9の新規打鍵ではない。A1二指、A3未学習対照の再評価は行わず、過去の評価を保持した。

## 否定的結果と次の作業

**5ジョブの表示上の実行環境は一致していたが、元R1/D3/D5が同じCPUモデル・同じ演算経路で実行されたという証拠は得られなかった。** 同じrunner image、Python、PyTorch、スレッド設定は、計算結果の厳密一致を保証しない。D8のCPU命令経路差がD3/D5の再現不一致を引き起こしたとする証拠も得られない。

次は新たな事前計画で、**同じ初期重み・同じ観測に対するネットワークの中間計算（各Linear、活性化、softmax、Entropy）**を固定して比較する。これにより数値差がどの演算で最初に現れるかを調べる。過去の学習を再実行せず、CPU環境が違う可能性を前提にしても原因を決めつけない。

D3のR2の歴史的勾配は未検証。教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002は引き続き未証明。元PR #34/#37〜#51には書き込まず、**自動マージ・Draft解除は禁止**。
