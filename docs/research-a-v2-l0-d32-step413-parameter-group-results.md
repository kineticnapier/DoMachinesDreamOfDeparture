# 研究A v2-L0-D32 — step413の更新前後パラメータ群交換（結果）

**判定：`valid_step413_parameter_group_swap`（技術的SUCCESS）。低速打鍵の学習改善は未確認。** D28の同一観測でPPO更新前後の`shared`/ `actor`重みを交差交換すると、左指押下確率の+0.121641159パーセントポイントは、共有層と行動出力層の**両方**の重み変更に対応した。

## 事前登録・原本
- **実装・テスト・解析より先**に[事前登録コミット `4beb8d0d`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/4beb8d0d959339a3588c9a8006ea247ce684d390)で[固定プロトコル](research-a-v2-l0-d32-step413-parameter-group-protocol.md)を保存。結果から条件・成立判定・停止条件を変更していない。
- 独立[Draft PR #75](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/75)、base D31研究ブランチ。監査PR #34、基準点PR #44、過去研究PR、元モデル・PPO・身体・報酬を変更していない。
- D8 [Actions #41397065687](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41397065687) / artifact #13119278298（初期重み）、D12 [Actions #41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292) / artifact #13130809065（更新後native重みと元の凍結バッチ識別）、D28 [Actions #41562899406](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41562899406) / artifact #13197677066（step413の8次元観測と保存済み確率）を読み取り専用で再利用。
- A0単指・seed320403・step413（4.130秒）、D27の物理介入で影響が確認された4.327秒のKeyDownに対応。D28の更新前後観測・Motor事前状態・診断状態は完全一致。

## 実測：固定観測での交差交換

`00`は両群更新前、`10`は共有層だけ更新後、`01`は行動出力層だけ更新後、`11`は両群更新後。criticは4条件とも更新前に固定。

| `shared` / `actor` | 左指押下確率 `P(left)+P(both)` | 00との差 |
|---|---:|---:|
| 00（更新前） | **0.381405845284462** | 0 |
| 10（sharedのみ更新） | **0.381942614912987** | +0.000536769629 |
| 01（actorのみ更新） | **0.382080629467964** | +0.000674784184 |
| 11（両方更新） | **0.382622256875038** | **+0.001216411591** |

- 更新前後の全4行動確率はD28の保存値と**各成分完全一致（最大絶対差0.0）**。argmaxは全条件`none`。
- 3群の更新前後重み差：`shared` 1312/1344要素、`actor` 132/132要素、`critic` 33/33要素。計1509要素。初期重みSHA256 `5420d95edaeea137bd7e6f0d5fc84c3890c41236f43e4cc4c44bd0392142f83c`、凍結バッチSHA256 `53976979540c9cbaf358c32175742df2616eaa16f0349fccda3028b3098c4f51`。
- `11`と`111`（criticだけ更新後に追加）は**logits・確率がビット単位で完全一致**。critic出力層は固定観測の方策logitsに直接入らない。
- 2群の対称平均寄与（更新順の両順序を平均）：
  - `shared` **+0.000539198518**（+0.053919852パーセントポイント、総差の44.327%）。
  - `actor` **+0.000677213073**（+0.067721307パーセントポイント、総差の55.673%）。
  - 交互作用 `p11-p10-p01+p00` = **+0.000004857779**（+0.000485778ポイント）。
  - 寄与の和と総差の再構成誤差 **0.0**。
- **注意**：対称平均は混成モデルの局所的な重み交換の寄与配分であり、PPOの行動別勾配・報酬・Adamの更新原因を示すものではない。critic層だけを交換してもlogitsが変わらないことは、critic損失が学習中のshared層更新に影響しないことを意味しない。

## 否定的結果と未解決事項
- **actorだけでもsharedだけでも更新前後の全確率差を再現しない。** 更新前後の確率差は一方の出力層だけの変更では説明できない。critic-onlyの直接効果は0だった。
- D19非介入sampled576目標：Hit **5→4**、Extra **44→46**、実際の物理KeyDown **49→50**。**学習改善なし**。今回の追加学習0、PPO/Adam更新0、勾配計算0、Motor再生0、物理KeyDown生成0、Hit/Extra新規採点0。方策推論は固定観測への**5回のみ**。
- D17の行動別PPO勾配（none/right/left/both）が今回のshared/actor群変更のどちらを引き起こしたか、value/entropyの寄与、Adam内部状態の因果寄与は未検証。1観測・1seedの混成モデルなので、閉ループのExtra期待値、A1二指、A3未学習対照、他seed、長期学習への一般化不可。
- 教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。

## 実装・検査・記録
- 実装：`scripts/a002v2_l0_d32_step413_parameter_group.py`、検査：`tests/test_research_a_v2_l0_d32_step413_parameter_group.py`、専用Actions：`.github/workflows/research-a-v2-l0-d32-parameter-groups.yml`。
- [初回Actions #41586834127 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41586834127)：**20 passed、警告1件**（NumPy任意依存未導入）。原本3件取得・SHA照合・5条件推論・陰性対照・結果保存が全て成功。
- [結果JSON・tests.log・run.log artifact #13207574466](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41586834127/artifacts/13207574466)（30日保持）。ZIP SHA256 `63cb76bd00914f94961b6500f40a0cef5e1d896219a89a28bf005fab7a288cc4`。
- この結果文書を含む最終HEADのActionsを再検証し、失敗なら事前登録・過去測定値を変更せず失敗を追記する。

**Draft維持。自動マージ・Draft解除・既存PRのマージは永久禁止。**
