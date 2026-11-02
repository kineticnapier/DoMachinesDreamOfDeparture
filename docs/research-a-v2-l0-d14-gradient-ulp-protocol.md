# 研究A v2-L0-D14 — 凍結PPO勾配のfloat32 ULP距離監査（実行前固定）

**事前登録：本書を先にコミットし、独立Draft PRに固定してから解析コード・検査・Actionsを追加する。** 基準点は研究A v2-L0-R2 Draft PR #44。正式研究002は未承認。リファクタリング監査PR #34とは独立。

## 単一の検証因子と問い
D13ではD12の同一バッチ・同一初期重みについて最初のAdam.step直前から292/1509勾配要素がCPU演算経路で異なることを確認した。今回は**その差のfloat32 ULP距離（隣接する表現可能数の段数）**だけを調べる。差がすべて1 ULP以内か、複数ULPに及ぶか、正負の反転やゼロとの出入りを伴うかを区別する。原因演算の特定や運指獲得の検証ではない。

## 固定入力・比較対象
- D12最終成功Actions #41424435292、artifact `research-a-v2-l0-d12-frozen-ppo-cpu`（ID 13130809065）の `capture.json`、`native.json`、`default.json`、`comparison.json` を**読み取り専用**で使用。追加学習・再学習・PPO更新・Motor実行をしない。
- D13の `validate_archived` と `compare_parameters` をそのまま呼び、元D12のモデル・バッチSHA、CPU環境、128目標・Hit0・Extra9・物理KeyDown9、Adam step前の差292/407要素、最大絶対差、最初の相違を厳密照合する。不一致は `invalid_archived_input` として停止。
- 各float32の符号付き32-bit生ビットを単調順序に変換して `abs(ordered_native - ordered_default)` をULP距離と定義。正負ゼロの生ビット差も1として数える。NaN/Inf、ビット範囲外、パラメータ名・長さ不一致は無効。
- Adam step 1直前・step 2直前の各1509要素を全比較し、差要素数、ULP距離の度数（1、2、3–4、5–16、17以上）、最大ULP、ULP>1要素数、符号反転（両方非ゼロで正負逆）、ゼロ↔非ゼロ、正負ゼロのみの差、パラメータ別の差数・最大ULP、最初の差をJSONに保存。ゼロ差も保存。
- `all_differences_within_one_ulp` は両段階の相違すべてが1 ULPの場合、`multi_ulp_difference_observed` はどちらかで2 ULP以上の場合、`no_gradient_difference` は両段階で差0の場合。D13の292/407との差を必ず検査し、今回の固定データでは `no_gradient_difference` も整合不良として停止する。事前に分類を変更しない。

## 停止条件と評価
- artifactが取得不能、元D12のSHA/物理記録/実行条件が不一致、D13の比較件数・最大絶対差・最初の差と不一致、非有限値、計測欠落の場合は解析を中止し、失敗ログと `invalid_archived_input` を保存する。
- 成功はULP分布を正しく保存することのみ。Hit率、左右交代率、指使用率を成功条件にしない。結果が1 ULP以内でも「単なる丸めが原因」とは断定しない。2 ULP以上でも数値差が学習失敗を引き起こしたとは断定しない。
- 新規学習0試行、PPO更新0回、Motor実行0、新規物理KeyDown0。A0 seed0の既存128目標のみ。A1二指・A3未学習対照・介入の追加実行なし。
- 過去D3/R2の歴史的勾配、D5/R1の再現失敗原因、PPO勾配差の最初の演算、教師なし運指獲得、単指の物理的不可能性、人間同等性、正式研究002の承認は未証明。

## 作業境界
D13 Draft PR #56のheadをbaseに新規ブランチ・**Draft PR**を作成する。元PR #34/#37〜#56、身体、報酬、PPO、学習済み重み、過去の結果・事前登録は変更しない。成功・失敗のActions artifactを保存し、結果文書は別コミットに保存する。自動マージ・Draft解除・既存PRのマージは永久禁止。
