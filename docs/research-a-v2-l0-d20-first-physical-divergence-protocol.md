# 研究A v2-L0-D20 — D19保存済み物理KeyDownの最初の分岐（実行前固定）

**状態：予備診断の事前登録。正式研究002の承認ではない。** 本書を解析コード・検査・Actions・結果より先に独立コミットし、結果を見た後に条件・停止基準・評価指標を変更しない。D19 Draft PR #62のhead `research/a-v2-l0-d19-paired-motor-post-ppo` を基点とする独立ブランチ/Draft PRでのみ作業。基準点 #44、監査 #34、D19および他の既存PRには書き込まない。

## 問いと単一因子
D19のA0単指・seed0における**初回PPO更新前後の保存済み同一条件Motor評価**で、物理KeyDown列が最初にどのイベントで分岐したか。比較因子はD19で固定された **before / after のモデル重み**だけ。新しいMotor評価・学習・PPO/Adam更新・CPU条件変更・報酬変更・身体変更は行わない。原因推定ではなく保存イベントの記述的解析である。

## 入力・比較対象
- **唯一の入力**：D19最終成功 Actions #41485667328、artifact ID **13161377329**、名前 `research-a-v2-l0-d19-paired-physical-motor` の `result.json`。元commit `08d8de523ffae10513645dfe95566060b4e6bf0d`。読み取り専用で使用。D19の全体結果を再計算せず、再学習しない。
- A0 argmax **9対/144目標**、sampled **36対/576目標**。同一 case_index/repetition/sampling_seed、16目標/試行、既存9条件。陰性対照はD19の `negative_control_exact` を確認し、再実行しない。
- 元D19の既知の照合値：sampled before Hit5/Extra44/物理KeyDown49、after Hit4/Extra46/物理KeyDown50。argmax両側0/0/0。D19学習時の過去記録128目標/Hit0/Extra9/物理KeyDown9は別物として扱う。
- 物理traceの各要素の `time_s`・`key`・`hit`・`closest_target_ordinal`・`offset_ms`・`immediate_reward` を比較。traceを**インデックス順に**照合し、最初に一致しない要素（片側の欠落も含む）を各対の「最初の分岐」とする。時間が最も早い分岐対も記録するが、同じインデックスが同じ因果的な押下に対応するとは仮定しない。
- 出力は全45対の分岐有無、sampled/argmaxの一致対数、分岐対の case/repetition/seed、最初の不一致位置、両側の物理イベント詳細、各対のHit/Extra/KeyDown差、Hit減少対とExtra増加対の重なり、分岐前の共通イベント数、集計値を保存。イベントがない側はnull。**行動指令・直前観測はD19に保存されていないため、推測して埋めない。**

## 停止条件・評価基準
- 入力の欠落、想定外schema/classification、9/36対の不足、case/repetition/seed不一致、陰性対照不一致、trace長・Hit/Extra/物理KeyDown計数不一致、非有限時刻、既知のD19集計値との不一致、イベントの形式不正、前後モデル・原本の上書き、実行失敗の場合は **`invalid_archived_trace_comparison`** とし、成功としない。失敗ログ・可能な部分結果を保存。
- すべてを満たせば **`valid_archived_trace_comparison`**。Hit改善は**sampled Hit増加かつExtra非増加**のときのみ真とし、D19の判定を再解釈しない。分岐が0対でも妥当な否定的結果として記録する。分岐の発見は学習成功でも因果関係の証明でもない。
- **新規学習0、PPO更新0、Adam更新0、Motor評価0、新規物理KeyDown0**。出力JSON、専用テスト、実行ログをGitHub Actionsで成功・失敗にかかわらずartifact保存。元D19 artifactは上書きしない。
- A1二指・A3未学習対照・身体介入は実施しない。左右交代率や使用率は成功条件にしない。人間同等性、単指物理的不可能性、正式研究002の承認は未証明。**マージ・Draft解除・自動マージは永久禁止。**
