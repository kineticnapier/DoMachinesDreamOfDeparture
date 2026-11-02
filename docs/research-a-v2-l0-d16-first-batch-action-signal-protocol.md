# 研究A v2-L0-D16 — 初回PPOバッチの行動別学習信号（実行前固定）

**事前登録。** D15 Draft PR #58 のHEADから独立ブランチ/Draft PRを作る。基準点はR2 Draft PR #44。DMDODリファクタリング監査PR #34とは独立。正式研究002は未承認。本書は解析実装・結果取得より先に別コミットで固定し、結果を見て判定を変更しない。

## 単一の問い
D12の**初回A0単指seed0・8試行の凍結学習バッチ**において、元PPOが用いる標準化済みadvantageと最初のepochのclipped policy objectiveが、選択された4行動（none=0/right=1/left=2/both=3）にどのように分布するかを測定する。D2の「学習後の評価軌跡に対する事後推定」と異なり、これは実際の**初回学習バッチ**を用いる。ただし選択行動ごとの寄与は勾配の因果的な分離でも、実際の物理KeyDownとの一対一対応でもない。

## 入力と固定条件
- D12最終成功Actions #41424435292 の artifact `research-a-v2-l0-d12-frozen-ppo-cpu` にある `frozen-batch.pt` / `capture.json` / `native.json` / `default.json` / `comparison.json` を読み取り専用で再利用。D13の既存検証で記録・CPU条件・勾配差を照合し、`d12.batch_digest` でバッチSHAを検証。
- D8成功Actions #41397065687 の `native-initial-state.pt` / `native.json` / `default.json` を再利用し、`d12.weights_digest` で元初期モデルSHAを照合。
- PyTorch 2.5.1+cpu、Python 3.11、単一スレッド、`ATEN_CPU_CAPABILITY`未設定（native）で**最初のepochのforwardのみ**。新規学習、backward、optimizer/Adam更新、Motor実行は0。元の身体・報酬・PPO・重み・バッチを変更しない。
- 元`update_ppo`と同じ全行の `adv=(returns-old_values)`、`(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)`、`Categorical(logits)`、`ratio=exp(log_prob-old_log_prob)`、`min(ratio*adv,clamp(ratio,0.8,1.2)*adv)` を使用。PPOのpolicy_lossは `-objective.mean()`。
- 4行動すべて（0〜3）を集計。行動選択数、advの正/負/ゼロ件数、advの平均・合計、objectiveの合計・平均、ratio範囲、clipped側がstrictly小さい件数を保存。全行のadv合計・objective合計と4行動別合計の整合を確認。元のforward計算と同一の全行policy_lossも記録。
- D12の過去の物理記録：A0単指・8試行・128目標、Hit0、Extra9、物理KeyDown9。新規打鍵0。A1/A3/介入は追加しない。左右交代率・使用率は成功条件にしない。

## 事前判定・停止
- `valid_action_signal_audit`：SHA、全行の有限値、4行動の数と合計、過去物理記録が一致し、各行動の指標を保存できた場合。符号や優劣に関係なくこの判定を採用。
- `invalid_audit`：入力欠落、SHA/物理記録不一致、異常な行動番号、非有限値、adv標準化・集計の不整合、想定外のモデル構造の場合。失敗ログ・部分出力を保存し、結果を成功と呼ばない。
- 行動ごとのadv合計が正でも、行動を強化した**実際の勾配方向**や物理打鍵の増加を証明しない。選択行動のadvとHit/Extraは直接対応させない。正の信号があっても運指獲得・低速学習の改善を主張しない。
- D3/D5の再現不一致の原因、CPU逆伝播の初出演算、A1/A3の学習信号、単指物理限界、人間同等性、正式研究002の承認は未証明として保持する。

## 保全
D15のブランチをbaseとする**別のDraft PR**に結果を保存。実行前に本事前登録だけをコミットし、PRをDraftで作成した後に解析コード・テスト・専用Actionsを追加する。既存PR #34/#37〜#58、既存の事前登録・重み・成果物を上書きしない。重複する長時間学習は開始しない。**自動マージ、Draft解除、マージを永久に禁止。**
