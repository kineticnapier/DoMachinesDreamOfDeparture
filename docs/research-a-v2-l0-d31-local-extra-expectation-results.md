# 研究A v2-L0-D31 — A0局所KeyDownの条件付き期待値（結果）

**判定：`valid_local_a0_keydown_expectation`（技術的検査成功）。低速打鍵の教師なし学習改善は未確認。** D27の「step413の1回のleft指令で4.327秒の余分なKeyDownが発生する」という局所介入と、D30のPPO更新前後の4行動確率を組み合わせ、同じ観測での**期待発生確率の差はわずか+0.0012164115905761719（+0.121641159パーセントポイント）**と計算された。

## 事前登録・原本・独立性
- [実行前固定のD31事前登録](research-a-v2-l0-d31-local-extra-expectation-protocol.md)：[コミット `cc30f95c`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/cc30f95c3d220123afd1e29049f0ba7edf057323)。解析実装・検査・計算前に固定。条件・停止・成立判定は変更していない。
- [独立Draft PR #74](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/74)、base D30研究ブランチ。監査PR #34、基準点PR #44、過去の研究PR・モデル・PPO・Motor・報酬は未変更。
- D27最終[Actions #41556553776](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41556553776) / artifact #13195156627の`result.json`、D30最終[Actions #41574924385](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41574924385) / artifact #13202832575の`result.json`を直接取得。両者のD22原本run #41515190995 / artifact #13175252296、A0 seed320403、step413を照合。D29の960/960原本再生・同一乱数陽性対照をD30の保存済み検査結果として照合。

## 主要測定値

A0の実装では右指指令を常に−1.0に固定する。したがって`none/right`は物理的に同じ左指非押下指令、`left/both`は同じ左指押下指令になる。D27の2組の開ループ介入はどちらも、step413で左指指令を入れると4.327秒の物理KeyDownが発生し、除くと消える。陰性対照・介入前状態の一致も照合成功。**新たに4行動をMotor再生したわけではない**。

| 指標（同じstep413観測、保存済み後続指令を固定） | PPO更新前 | 更新後 |
|---|---:|---:|
| `P(none)+P(right)`：左指非押下指令 | 0.6185941547 | 0.6173777431 |
| `P(left)+P(both)`：左指押下指令 | **0.381405845284462** | **0.38262225687503815** |
| 条件付き4.327秒KeyDown期待発生確率 | **38.1405845%** | **38.2622257%** |

- 期待発生確率の差：**+0.0012164115905761719（+0.12164115905761719パーセントポイント）**。
- 更新後/更新前の比：**1.0031892840805021**（約+0.319%の相対増加）。
- 固定したB側後続指令・A側後続指令の両条件で、同じ期待値と差を得た。
- D29の保存済み乱数で実際に`none→left`が発生した事実は維持される。しかし**その1回の実現値を、一般に大きな確率変化が起きた証拠として扱えない**。
- この値は1回の**物理KeyDownの期待増分**であり、未採点の介入系列の`Extra`数や報酬期待値ではない。

## 否定的結果と限界
- **step413の単一指令の局所因果効果が大きくても、PPO更新による同指令の選択確率の変化は約0.122ポイントにとどまった。** この1時点だけの計算でD19全36対のExtra増分+2を説明したとは言えない。
- D19非介入sampled576目標：Hit **5→4**、Extra **44→46**、実際の物理KeyDown **49→50**。**学習改善なし。**
- 今回の新規学習0、PPO/Adam更新0、方策推論0、Motor再生0、物理KeyDown生成0、Hit/Extra新規採点0。D27の介入系列は未採点。
- A0の固定観測1件、開ループ指令の条件付き期待値であり、閉ループの後続方策反応、他の観測、長期学習、A1二指、A3未学習対照への一般化は不可。**なぜPPOがこの確率方向に変えたかという学習信号の原因は未解決。**
- 人間同等性、単指物理的不可能性、教師なし運指獲得、正式研究002の承認は未証明。

## 実装・検証・記録
- [実装コミット `129b31d8`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/129b31d88eb2df06627337187e00204724eb5be5)：解析`scripts/a002v2_l0_d31_local_extra_expectation.py`、11件のテスト、専用Actions。原本・介入の双方向反転・A0指令同値・異常入力拒否を検査。
- [初回Actions #41580488341 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41580488341)：**11 passed**、原本2件取得成功、条件付き期待値計算成功。Node.js 20非推奨警告のみ。
- [結果JSON・テストログ・実行ログ artifact #13206025404](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41580488341/artifacts/13206025404)（30日保持）。
- 結果文書を含む最終HEADでも同じ専用Actionsを再検証する。失敗した場合は失敗を記録し、事前登録・測定条件を変更しない。

**Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
