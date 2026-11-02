# 研究A v2-L0-D8 — CPU演算命令の選択が学習前Entropyの厳密再現性に及ぼす影響（実行前固定）

**状態：計測コード・CI実行前の固定計画。正式研究002は未承認。** D3〜D7の結果から立てた診断仮説であり、盲検の新規研究とは呼ばない。

## 問いと操作する唯一の因子

D3のR1初期評価 `1.3561848871858129` とD7のR1初期評価 `1.3561848983669778` は、同じseed0・学習前でも厳密には一致しなかった。R2のD3再現失敗、R1のD5再現失敗は未解決。**CPU向けPyTorch演算命令の自動選択（ATen CPU capability）だけを変更すると、同じ初期重み・同じ評価入力でEntropyまたは物理打鍵の結果が変わるか**を検査する。

同じGitHub Actions実行環境で、**別々のPythonプロセス**を使って次の2条件を実行する。

- **native**: `ATEN_CPU_CAPABILITY` を設定しない（PyTorchの通常のCPU命令選択）。
- **default**: Pythonの起動前に `ATEN_CPU_CAPABILITY=default` を指定し、ATenの汎用CPU命令経路を選ぶ。

`torch.backends.cpu.get_cpu_capability()` とCPUモデルを記録し、native/defaultの実際の命令経路が同じなら **`dispatch_not_distinct`** と判定して原因の検証には使わない。これはPyTorch全体の演算経路を固定するものではない。環境変数が有効でも全演算がその指定に従うとは仮定しない。

## 固定入力と変更禁止事項

- 元R1 Actions #41309837474 / artifact #13084957586、D3 Actions #41369778494 / artifact #13109230783、D7 Actions #41391082538 / artifact #13116998596 を読み取り専用で参照する。
- 元R1/D3のR1 A0 seed0初期評価全行と初期重みSHA-256を照合し、同一の元モデルであることを確認する。D7の初期Entropyの既知値も比較対象として固定する。
- Python 3.11、PyTorch 2.5.1+cpu、`torch.set_num_threads(1)`、`PYTHONHASHSEED=0`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。CPUモデルとTorchの演算設定を記録する。
- 元の`ActorCritic`、元の`evaluate_set`、`torch.manual_seed(0)`で生成した**学習前モデル**を使用。A0単指・210/240/250ms × 開始3種、argmax9試行＋sampled36試行。元と同じ抽選種、同じ観測、同じ身体、同じ判定。
- **学習0試行・PPO更新0回**。報酬、Motor、観測、重み、PPO、教師運指、元R1/R2/D3〜D7の結果を変更しない。A1/A3の追加試行をしない（既存結果を参照するだけ）。

## 計測と事前判定

1. 事前に元R1とD3の初期モデルSHA・全初期評価を厳密照合。欠損・不一致なら実行前に `invalid_archives` で停止。
2. nativeとdefaultで学習前モデルを独立に生成し、初期重みのSHA-256と全`evaluate_set`結果（argmaxとsampled）を保存する。実行条件・CPU機能・Python/Torchの版・全評価行を保存。新しい出力を上書きしない。
3. 同じCPU・異なるATen命令経路・同じ初期重みを確認したうえで、両条件の初期評価を**厳密一致**で比較。最初の異なるフィールド、Entropyの値と差、Hit/Extra/物理KeyDownの一致・不一致を報告。
4. 元R1/D3の初期評価に対して両条件の差を厳密比較し、既知のD7 Entropy値との一致も報告。比較で元R1またはD7の値を再現しても、それだけでD3/D5の原因とは断定しない。
5. 判定は `dispatch_effect_observed`（同一重み・異なる命令経路で初期評価に差）、`no_dispatch_effect_observed`（異なる命令経路だが全評価一致）、`dispatch_not_distinct`（命令経路が異ならない）、`invalid_diagnostic`（同一初期重み・保存記録・検査が不成立）のいずれか。**成功のために許容誤差を導入しない**。
6. 変化がEntropyだけで物理KeyDownに変化がない場合もそのまま報告。元のR1/D3の完全再現・R2の歴史的勾配・運指獲得・人間同等性・単指物理限界・正式002を認証しない。

## 停止・保存・運用

学習なし、最大2条件×45評価試行で終了。既存CPU実験の同時実行があれば重複起動しない。条件・評価コードの検査に失敗した場合はCI失敗とし、ログと部分結果を可能な限り保存する。結果JSON・差分・検査ログを新規artifactへ保存し、否定的結果もPRに残す。

この文書を**実験コード追加前**に新規ブランチへコミットし、別のDraft PRで隔離する。PR #34および#37〜#50は変更しない。**自動マージとDraft解除は永久禁止。**
