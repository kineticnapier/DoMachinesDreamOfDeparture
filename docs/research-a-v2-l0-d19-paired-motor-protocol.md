# 研究A v2-L0-D19 — 初回PPO更新後の実Motor対比較（実行前固定）

**事前登録：本書を実装・実行・結果より先に独立コミットする。** D18 Draft PR #61のHEAD `8ee75bc5668c96553e4719a45e58485a028666c3` から分岐し、D18をbaseとする新規Draft PRのみで作業する。基準点PR #44・監査PR #34・旧研究PRは変更しない。正式研究002は未承認。事後の条件・停止・判定基準変更は禁止。

## 単一の問い・因子
D12で**既に実施された最初の1 PPO更新（2 Adam step）**の前後で、A0単指の**同一Motor評価条件**におけるHit、Extra、実際の物理KeyDownがどう変わるか。因子は重みだけ（before: D8保存初期重み、after: D12保存native更新後重み）。CPU条件、身体、報酬、観測、評価seed、PPO実装は固定。学習・PPO/Adam更新は行わない。D18の凍結観測の確率変化を学習成功と読み替えない。

## 入力・条件
- D8成功Actions #41397065687 artifact `research-a-v2-l0-d8-cpu-dispatch` の `native-initial-state.pt`, `native.json`, `default.json`。D12成功Actions #41424435292 artifact `research-a-v2-l0-d12-frozen-ppo-cpu` の `frozen-batch.pt`, `capture.json`, `native.json`, `default.json`, `comparison.json`。D18の既存SHA/float32全重み照合とD12バッチSHA/物理記録検査を再利用。元ファイルは読み取り専用。D12のA0 seed0、8試行、128目標、Hit0、Extra9、物理KeyDown9は**学習時の過去記録**としてのみ報告する。
- 同一runner・Python 3.11・torch 2.5.1+cpu・1スレッド・ATEN_CPU_CAPABILITY未設定。両方のモデルを `eval()`、`torch.no_grad()` で元の `a002v2_l0_low_speed_ppo.evaluate_set` に渡す。
- A0単指の元L0評価格子を変更しない。間隔210/240/250ms × 開始0.835/1.075/1.290秒、argmax各1回=9試行/144目標、sampled各4回=36試行/576目標。sampled seedは元の `320000+100*case_index+repetition`。beforeとafterは全45試行で同じcase/repetition/seedを使う。**陰性対照**としてbeforeを同じ45条件でもう1回評価し、全物理trace・Hit・Extra・KeyDown・行動数の完全一致を要求。
- 主評価はsampledのHit/576、Extra/576、物理KeyDown/576および各36試行の対応差。argmaxのHit/144、Extra/144、物理KeyDown/144と9試行の対応差は補助評価。各物理KeyDownは時刻・指・Hit/Extraをtraceに保存。各行で `hits+extra_keydowns=physical_keydowns`、trace長一致、目標16を要求。物理イベントの増減を行動選択回数で代用しない。
- A3未学習一様乱択とA1二指の**新規評価は今回行わない**。比較対象不足を明記し、A0の初回更新だけから運指獲得を結論しない。左右交代率・使用率は成功基準にしない。

## 停止・分類（結果を見て変更しない）
- 入力欠落、SHA/ビット列不一致、保存済みD12履歴不一致、実行環境不一致、非有限、評価格子・seed・対応行の不一致、物理trace計数不一致、陰性対照不一致、実行失敗の場合 `invalid_paired_motor_evaluation` とし、成功扱いしない。失敗ログ・部分結果をartifactへ保存。
- 全検査を満たした場合、数値の増減にかかわらず `valid_paired_motor_evaluation` とする。別途 `sampled_hit_increase_without_extra_increase` は **after sampled Hit > before sampled Hit かつ after sampled Extra <= before sampled Extra** のときのみ真（それ以外は偽）。これは**限定された初回PPO更新の観測結果**であり、教師なし運指獲得の成功判定ではない。物理KeyDownの差は必ず報告する。
- 長時間研究ジョブが進行中なら重複起動しない。追加学習0、PPO更新0、Adam更新0。元モデル・身体・報酬・既存結果の上書き禁止。GitHub Actionsで結果JSON、検査・実行ログを成否にかかわらず保存。実験後の失敗を消さない。

## 解釈の制限
D3/D5の歴史的再現失敗、A1二指・A3未学習対照・介入、人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。**自動マージ・Draft解除・マージは永久禁止。**
