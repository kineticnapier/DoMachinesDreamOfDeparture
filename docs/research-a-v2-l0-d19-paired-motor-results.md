# 研究A v2-L0-D19 — 初回PPO更新前後のA0実Motor対比較（実測結果）

**判定：`valid_paired_motor_evaluation`。否定的結果：保存済み初回PPO更新後のA0 sampled Hitは5→4に減少し、Extraは44→46に増加。** 事前登録 `402f23c65db86bea0a6baaf35662c6b0ebe4f693` の条件・停止・判定基準は変更していない。新規学習0、PPO更新0、Adam更新0。実Motorシミュレータの評価のみ。

## 実行・保全
- [事前登録](research-a-v2-l0-d19-paired-motor-protocol.md) は実装より先に独立コミットで固定。D18のHEAD `8ee75bc5` を基点とする独立Draft PR #62（base D18 PR #61）。
- [成功GitHub Actions #41485429186](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41485429186)：**SUCCESS**。専用テスト **9 passed、NumPy任意依存の警告1件**。D8/D12保存成果物の取得とSHA照合、評価、結果保存がすべて成功。
- [artifact #13162091507](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41485429186/artifacts/13162091507)：`result.json`（全90件の前後評価行・物理KeyDown trace、45件の陰性対照の一致判定、対応差）、`tests.log`、`run.log`。30日保存。原本D8/D12と過去研究結果は上書きしない。
- Python 3.11.17、torch 2.5.1+cpu、1スレッド、native CPU経路。A0 seed0のD8初期重みとD12の**過去に2 Adam step更新したnative重み**を比較。学習データはD12の8試行/128目標/Hit0/Extra9/物理KeyDown9を照合し、学習は再実行していない。

## A0同一条件のMotor評価
元L0の9条件（間隔210/240/250ms×開始0.835/1.075/1.290秒）と固定seedを変更せず、beforeとafterを対比較。実際にMotorが生成したKeyDownを計測し、行動指令数と混同しない。

| 指標 | before | after | after−before |
|---|---:|---:|---:|
| **sampled Hit / 576目標** | **5** (0.8681%) | **4** (0.6944%) | **−1** |
| **sampled Extra / 576目標** | **44** (7.6389%) | **46** (7.9861%) | **+2** |
| **sampled 物理KeyDown** | **49** | **50** | **+1** |
| argmax Hit / 144目標 | 0 | 0 | 0 |
| argmax Extra / 144目標 | 0 | 0 | 0 |
| argmax 物理KeyDown | 0 | 0 | 0 |

- sampledは36組中、Hit減少が**1組**（増加0組）、Extra増加が**1組**（減少0組）、物理KeyDown増加が**1組**（減少0組）。各行の物理KeyDown時刻・左右・Hit/Extraはartifactに保存。
- argmaxは9組すべてで物理KeyDown0、Hit0。初回PPO更新による打鍵獲得は観測できなかった。
- beforeを同じ全45条件で再評価した**陰性対照は全結果完全一致**。各行でHit+Extra=物理KeyDown、trace長=物理KeyDown、目標16、同一seedを検査。
- 新規Motor評価はbefore45 + 陰性対照45 + after45 = **135試行**。これらで生成された物理KeyDownイベントは **148件**（before49 + 陰性対照49 + after50）。これは**シミュレータ内の物理イベント**であり、本家ADOFAIの実機打鍵ではない。
- 事前登録した `sampled_hit_increase_without_extra_increase` は **false**。数値の符号にかかわらず比較自体の成立判定 `valid_paired_motor_evaluation` は維持。

## 否定的結果と解釈の限界
- D18で更新前後の固定3,800観測のargmaxが277件変化しても、今回の**実際のMotor評価**ではHit改善を確認できなかった。sampledはHit減少・Extra増加、argmaxは物理KeyDown0のまま。確率変化を物理打鍵改善と同一視する仮説は支持されない。
- この結果は**初回8学習試行後の1 PPO更新だけ**のA0 seed0比較であり、長期学習の否定でも教師なし運指獲得の達成でもない。A1二指、A3未学習対照、身体介入は今回未実施。比較対象不足を明記する。
- D3/D5歴史的再現失敗の因果関係、人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。左右交代率・使用率は成功基準にしていない。
- 実装とテストは新規D19専用ファイルのみ。身体・報酬・PPO本体・元モデル・旧研究PR・監査PR #34・基準点PR #44は未変更。**Draft維持、自動マージ・Draft解除・マージは永久禁止。**

## 次の検証候補（今回未実行）
更新前後で差が生じた**対応評価行の物理KeyDown時刻と指・判定・直前観測**を既存D19 `result.json` から抽出し、Hit減少/Extra増加の直接の分岐を一因子で調査する。まず保存結果の読み取りのみで行い、新規学習はしない。別の事前登録とDraft PRを作るまで実行しない。
