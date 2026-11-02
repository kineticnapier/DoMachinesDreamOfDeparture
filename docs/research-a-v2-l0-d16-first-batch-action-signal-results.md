# 研究A v2-L0-D16 — 初回PPOバッチの行動別学習信号（実測結果）

**判定：`valid_action_signal_audit`。** 保存済みの最初のA0単指・seed0のPPOバッチについて、行動別の標準化advantageと初回policy objectiveを全3,800行で測定した。**追加学習0、backward0、PPO更新0、Adam更新0、Motor0、新規物理KeyDown0。**

## 事前登録・実行・保存
- [実装前の事前登録](research-a-v2-l0-d16-first-batch-action-signal-protocol.md) はコミット [`4d1987df`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/4d1987df9679ff4a46597de5e0a9186f78cafc31) に固定。結果を見て基準を変更していない。
- [Draft PR #59](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/59) はD15 Draft PR #58をbaseにした別ブランチ。監査PR #34は未変更。
- D12成功 [Actions #41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292) の凍結バッチ・検査済み記録と、D8成功 #41397065687 の初期モデルをSHA照合して再利用した。
- [D16 Actions #41458427405](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41458427405) は**SUCCESS、8テスト成功・警告1件**。[全3,800行のJSON・実行ログ・検査ログ artifact #13147183760](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41458427405/artifacts/13147183760)。

## 実測：選択行動別の初回PPO学習信号

| 行動 | 選択数 | 正のadv | 負のadv | 平均adv | policy objective合計 | PPOクリップ側採用 |
|---|---:|---:|---:|---:|---:|---:|
| none（0） | 1,204 | 527 | 677 | -0.00104008 | -1.25224352 | 0 |
| right（1） | 1,170 | 497 | 673 | -0.02103906 | -24.61569709 | 0 |
| left（2） | 783 | 352 | 431 | **+0.06227274** | **+48.75955612** | 0 |
| both（3） | 643 | 266 | 377 | -0.03560134 | -22.89166088 | 0 |
| **合計** | **3,800** | **1,642** | **2,158** | — | 約0 | **0** |

- 行動別のadv平均は、**leftのみ正、none/right/bothは負**。leftも783選択中431件は負であり、単純な「leftを押せば成功」の信号ではない。
- 初回forwardでの比率 `exp(log_prob - old_log_prob)` は全行ほぼ1（約0.99999964〜1.00000024）で、元PPOの0.8〜1.2のクリップ境界に達した行は**0件**。
- 初回の `policy_loss` は **1.2360120926757645e-08**。標準化advantageの全体平均がほぼ0のため、この値だけから「学習していない」とは判定できない。
- これは**実際の初回学習バッチ**の選択行動別の損失項。行動別の平均advや損失寄与は、逆伝播後の各行動の勾配方向を直接表さない。行動と物理KeyDownの因果的な一対一対応も計測していない。
- A0単指ではright指の物理接点が無効でもright/bothという**指令**は選択可能。指令の選択数を物理KeyDown数と混同しない。

## 否定的結果・身体測定
D12保存済みのA0 seed0・8試行・**128目標に対してHit0、Extra9、物理KeyDown9**。今回の新規打鍵は0。初回バッチの行動別信号を観測できても、**低速打鍵の学習獲得・Hit改善は示されない**。初回のPPOクリップ境界も発動していない。

A1二指、A3未学習対照、指への介入は追加していない。左右交代率や指使用率を成功条件にしていない。D2の学習後評価に対する事後advantageとは測定対象が異なる。

## 未解決・保全
行動別の**実際の方策勾配方向**、各指令が物理KeyDownになるまでの因果的寄与、後続バッチ・A1/A3での信号、D3/D5再現不一致、CPU逆伝播の初出演算は未検証。教師なし運指獲得、単指の物理的不可能性、人間同等性、正式研究002の承認も未証明。

既存研究PR #37〜#58、基準点 #44、監査PR #34、身体・報酬・PPO本体・元重み・元結果を変更していない。**Draft維持、マージ・Draft解除・自動マージは禁止。**
