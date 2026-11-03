# 研究A v2-L0-D30 — 同一乱数seed 4096件でのstep413抽選感度（実測結果）

**判定：`valid_seed_sensitivity_audit`（技術的検査成功）。低速打鍵の学習改善は確認できない。** D29で見つかった `none→left` は、固定した4096個の初期seedでは**0件**だった。抽選結果が異なったのは**6/4096件（0.146484375%）**であり、6件すべて更新前の `right` からの変化だった。これはD29の1つの乱数状態における反例を否定するものではなく、微小な確率差による抽選分岐が乱数状態に依存することを示す。

## 実行前の固定と監査との独立性

- [事前登録](research-a-v2-l0-d30-common-rng-seed-sensitivity-protocol.md)は解析コード・検査・実測前に[コミット `5b80419a`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/5b80419a662dc6e9df1a979ac40b1df5667472e6)として固定。比較条件、seed集合、停止基準、成立判定は変更していない。
- [独立Draft PR #73](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/73)はD29 Draft PR #72のHEADを起点とし、baseはD29研究ブランチ。基準点PR #44、監査PR #34、既存研究PRは変更していない。
- A0単指・seed320403・240ms間隔・開始1.075秒・16目標・480制御stepのD22保存記録を再利用。[D22原本Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995) / artifact #13175252296。D29の原本検証、陰性対照、B/A各480抽選の**960/960一致**、step413直前の乱数状態一致、既知の `none(0)→left(2)` を再確認。
- Python 3.11、torch 2.5.1+cpu、1スレッド、`ATEN_CPU_CAPABILITY`未設定。更新前後の保存済み4行動確率をfloat32で復元し、seed **0〜4095** を事前指定。各seedに同じ初期乱数状態の2生成器を用意し、各確率分布で `torch.multinomial` を1回ずつ実行。合計**8192抽選**。事後のseed追加や条件変更はしていない。

## 主要測定値

保存されたstep413の行動確率（行動順：none, right, left, both）：

- PPO更新前B：`[0.3233119547367096, 0.2952822148799896, 0.21348761022090912, 0.16791823506355286]`
- 更新後A→none：`[0.32353976368904114, 0.29383787512779236, 0.21458901464939117, 0.16803324222564697]`
- 確率分布の全変動距離：**0.0014442801475524902**。この値を、今回の同一seed抽選結合の不一致率の上限と**解釈しない**。

更新前を行、更新後を列とする抽選行動の遷移件数：

| 更新前＼更新後 | none | right | left | both | 行合計 |
|---|---:|---:|---:|---:|---:|
| none | 1344 | 0 | 0 | 0 | 1344 |
| right | **3** | 1209 | **3** | 0 | 1215 |
| left | 0 | 0 | 848 | 0 | 848 |
| both | 0 | 0 | 0 | 689 | 689 |
| 列合計 | 1347 | 1209 | 851 | 689 | **4096** |

- 不一致：**6/4096 = 0.00146484375（0.146484375%）**。一致：4090/4096。
- `right→none`：seed **925, 1050, 3943** の3件。
- `right→left`：seed **294, 797, 1818** の3件。
- **`none→left`：0件、`left→none`：0件。** D29での元のstep413乱数状態の `none→left` は4096件の集計に含めない。
- 4096件の初期乱数状態はそれぞれの対で完全一致。遷移件数合計4096、主要値有限、同一条件の再実行結果完全一致。判定 `valid_seed_sensitivity_audit`。

## 否定的結果と解釈の限界

- **「D29の `none→left` が一般的な抽選変化である」という推測は、この固定4096初期seedでは支持されない。** ただし4096seedは有限の固定集合であり、乱数状態全体の確率的な推定や保証ではない。今回の同一seed抽選は、実際の長い抽選履歴の途中の乱数状態と同一分布だとも断定しない。
- D29の既知の保存状態では、同じ乱数状態で確率だけを変えると `none→left` となる。今回の0件はその局所的な因果再生と矛盾しない。D27で確認した余分な物理KeyDownも取り消さない。
- D19の**非介入sampled評価576目標**：Hit **5→4**、Extra **44→46**、物理KeyDown **49→50**（PPO更新前→後）。**学習改善なし。** 今回は新規Hit率・Extra・物理打鍵を測定していない。
- 新規学習0、PPO/Adam更新0、方策再推論0、Motor再生0、新規物理KeyDown0、新規Hit/Extra採点0。左右交代率・使用率を成功条件にしていない。
- A1二指・A3未学習対照・他の物理条件・他のPyTorch/CPU環境・長期学習への一般化は未検証。**PPOがこの方向に行動確率を変えた理由は未解決。** 教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。

## 検査・記録

- [初回Actions #41574519461 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41574519461)：**9 passed、1 warning**。4096seedの8192抽選、D29原本960/960再生、既知陽性対照が成立。結果JSON・検査ログ・実行ログは[artifact #13202562488](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41574519461/artifacts/13202562488)（30日保持）に保存。PyTorchのNumPy初期化警告とNode.js非推奨警告は測定失敗ではない。
- 専用コード `scripts/a002v2_l0_d30_common_rng_seed_sensitivity.py`、専用検査 `tests/test_research_a_v2_l0_d30_common_rng_seed_sensitivity.py`、専用Actions `.github/workflows/research-a-v2-l0-d30-common-rng.yml`。成功・失敗ログはActions artifactに保持。
- 技術的な抽選検査の成立は運指獲得の成功を意味しない。**Draft維持、自動マージ・Draft解除・既存PRマージは永久禁止。**
