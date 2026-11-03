# 研究A v2-L0-D33 — 初回PPO行動別勾配のstep413左指確率への方向微分（結果）

**技術判定：`valid_step413_gradient_projection`（SUCCESS）。17 tests passed / 1 warning。低速打鍵の学習改善なし。**

## 実行前固定・原本
- [事前登録コミット `bb2be827`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/bb2be827b6a3e05b7f3adc3ac5fb172535315b51) と[固定プロトコル](research-a-v2-l0-d33-step413-gradient-projection-protocol.md)を**解析コード・テスト・実測より前**に保存。事後変更なし。
- [独立Draft PR #76](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/76)、baseはD32研究ブランチ。監査PR #34、基準点PR #44、過去研究PRは未変更。
- 原本：D8 [Actions #41397065687](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41397065687)、D12 [#41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292)、D17 [#41472753103](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41472753103)、D28 [#41562899406](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41562899406)。保存済みSHA・D17行動別勾配・D28観測/確率を照合。
- 初期重みSHA256 `5420d95edaeea137bd7e6f0d5fc84c3890c41236f43e4cc4c44bd0392142f83c`、凍結バッチSHA256 `53976979540c9cbaf358c32175742df2616eaa16f0349fccda3028b3098c4f51`。

## 実測：初期重みでの微小SGD方向微分
`p=P(left)+P(both)`、`s=-∇p·∇L`。**単位SGD学習率あたりの一次近似**であり、実際のAdam更新の確率差ではない。符号が正なら当該損失だけの微小勾配降下はpを増やす方向、負なら減らす方向。

| 凍結初回PPOの損失成分 | shared方向 | actor方向 | critic方向 | 合計方向微分 |
|---|---:|---:|---:|---:|
| none (1204行) | -0.0043292934 | -0.0037129971 | 0 | **-0.0080422905** |
| right (1170行) | -0.0016425664 | -0.0010879704 | 0 | **-0.0027305368** |
| left (783行) | +0.0064228296 | +0.0068875957 | 0 | **+0.0133104253** |
| both (643行) | +0.0029899995 | +0.0020509123 | 0 | **+0.0050409118** |
| 方策損失合計 | +0.0034409692 | +0.0041375403 | 0 | **+0.0075785095** |
| value+entropy補助 | -0.0010636129 | +0.0003460049 | 0 | **-0.0007176080** |
| PPO損失全体 | +0.0023773553 | +0.0044835452 | 0 | **+0.0068609006** |

- 行動別の方策方向は、**left/bothが正、none/rightが負**。value+entropy補助は合計で**負**。つまり初回PPO勾配方向の合計は正だが、全成分が左指確率を増やす方向だったわけではない。
- 方策方向の再構成誤差 **3.6620e-10**、全PPO方向の再構成誤差 **1.3059e-9**。事前基準1e-7以内。
- D17勾配再構成誤差：方策 **1.90485e-8**、全PPO **2.98023e-8**（事前基準1e-6以内）。D17保存済みの行動別ノルム/件数との照合成立。固定step413の4確率とD28保存値の最大差 **0**。
- 初期重みの固定step413 `p` はfloat32合計 **0.3814058304**。D28の保存済み4確率をfloat64加算すると **0.3814058453**。合計方法の丸め差であり、各行動確率の照合差は0。
- D28/D32で保存された実際のAdam 2 step後の確率差は **+0.001216411590576**（+0.121641159ポイント）。**今回の+0.0068609006をその実測差として解釈してはならない**。微小SGDとAdam 2 stepは別の更新則。

## 否定的結果・限界
- 「すべての行動別学習信号が左指確率を増やす」という説明は**否定**。none/rightは逆向き、補助損失も合計で逆向き。
- D17の平均advの符号、行動別勾配ノルムだけでは局所確率方向を判定できない。今回の投影は、**更新後重みの実際の原因を行動別に分解したものではない**。Adamの内部状態、2回目の勾配、更新非線形性は未検証。
- D12凍結8試行/128目標はHit **0**、Extra **9**、実際の物理KeyDown **9**。D19非介入576目標はHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。**低速打鍵の学習改善なし**。
- 新規学習0、PPO更新0、Adam更新0、Motor再生0、新規物理KeyDown0、Hit/Extra新規採点0。勾配計算8回、元重みビット不変、critic-only `∇p` は0。
- A1二指・A3未学習対照・他seed・閉ループ一般化、人間同等性、単指物理的不可能性、教師なし運指獲得、正式研究002承認は**未証明**。

## 実装・検査・保存
- 実装 `scripts/a002v2_l0_d33_step413_gradient_projection.py`、検査 `tests/test_research_a_v2_l0_d33_step413_gradient_projection.py`、専用Actions `.github/workflows/research-a-v2-l0-d33-gradient-projection.yml`。
- [初回Actions #41592989639 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41592989639)：**17 passed / 1 warning**（任意NumPy未導入）。4件の原本取得、SHA照合、8勾配、陰性対照、結果JSON保存成功。
- [初回結果JSON・tests.log・run.log artifact #13210188785](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41592989639/artifacts/13210188785)（保持30日）。
- 結果文書を含む最終HEADのActionsを再確認し、失敗時は事前登録・既存結果を上書きせず追記する。

**Draft維持。自動マージ・Draft解除・既存PRのマージは永久禁止。**
