# 研究A v2-L0-D13 — 保存済みPPO勾配の最初の分岐段階（実行前固定）

**事前登録：この文書を先にコミットし、Draft PRに固定してから解析コード・検査・Actionsを作る。** 研究A v2-L0-R2 #44は継続Draft、正式研究002は未承認。DMDOD監査PR #34とは別作業。

## 単一の検証因子
D12の同じ凍結バッチで生じたCPU演算経路差が、**1回目のAdam更新前の勾配から存在するか、2回目の更新前に初めて現れるか**を区別する。CPU差の因果的起源や学習失敗の原因は、この事後解析だけでは特定しない。

## 入力・比較対象
- D12最終SUCCESS Actions #41424435292 のartifact `research-a-v2-l0-d12-frozen-ppo-cpu`（ID 13130809065）の `capture.json`, `native.json`, `default.json`, `comparison.json` を読み取り専用で使用する。過去のモデルや勾配を作り直さない。
- D12の事前登録・結果、D12の両CPU実行が同じ初期重みSHAとバッチSHAを使用したこと、CPU capabilityが異なること、同じCPU/OS/Python/PyTorch/1スレッドであることを再確認する。元D12の `ppo_update_difference_observed`、Hit0/128、Extra9、物理KeyDown9、更新後重み差10、Adam exp_avg差572・exp_avg_sq差739、返却損失4項目の差0を照合する。
- `actor_and_critic_gradients` の2回のAdam.step直前のfloat32ビット列を、native/default間で名前・形状・要素ごとに完全比較する。各段階の差要素数、最大絶対差、最初の差の名前/番号/両ビット値/float32値、差のあるパラメータ数を記録する。ゼロ差も保存する。
- 1回目の差が存在すれば `gradient_difference_before_first_adam_step`、1回目は一致し2回目だけ異なれば `gradient_difference_only_before_second_adam_step`、両方一致なら `no_gradient_difference_in_two_steps`。これらは因果結論ではない。

## 停止条件・評価基準
- D12 artifactを取得できない、元D12の記録が欠ける、初期重み・バッチSHA・CPU環境・capability・計測数・物理KeyDown・D12の保存済み比較結果が不整合、NaN/Inf、整数ビット範囲やパラメータ名・長さ不一致なら `invalid_archived_input` として停止し、失敗ログを保存する。結果を見て基準を変更しない。
- 成功条件は**保存済み勾配の分岐段階を監査可能に記録すること**であり、Hit率改善・左右交代・指使用率ではない。追加学習0、PPO更新0、Motor実行0、新規物理KeyDown0。A0以外のA1/A3は今回追加しない。
- 過去D3/R2の歴史的勾配、D5/R1再現失敗の原因、CPU経路の最初の数値演算、PPO学習失敗の原因、単指の物理限界、人間同等性、正式研究002の承認は未証明として維持。

## 作業境界
D12 Draft PR #55のheadをbaseとする**新規の独立ブランチ/Draft PR**でのみ記録する。D12の過去成果、元重み・報酬・身体・PPO、監査PR #34、研究PR #37〜#55は変更しない。Actions artifactに成功・失敗の証拠を残し、結果文書を後続コミットに保存する。自動マージ・Draft解除・既存PRのマージは永久禁止。
