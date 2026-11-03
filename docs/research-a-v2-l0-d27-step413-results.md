# 研究A v2-L0-D27 — step413単一左指指令介入：測定結果

**判定：`valid_step413_single_command_intervention`（SUCCESS）。step413（4.130秒）の1回の有効左指指令を双方向に交換するだけで、4.327秒の余分な物理KeyDownが双方向で出現・消失した。** 低速打鍵の学習改善の証拠ではない。

## 実行前固定と分離
- 解析コード・テスト・実測前の[事前登録コミット `e20a4ea2`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/e20a4ea2c0e320f020b5823816b848988bc446fa)と[固定プロトコル](research-a-v2-l0-d27-step413-protocol.md)。事前登録の判定・条件は実測後に変更していない。
- 独立[Draft PR #70](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/70)。baseはD26 Draft #69のHEAD `39e6a29a`。監査PR #34・基準点PR #44・過去の研究PRは変更していない。
- D22成功[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995)の[artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296)の原本をD23検査で照合。A0単指・seed320403・240ms間隔・開始1.075秒・16目標・480制御step・10ms制御・1ms物理。新規学習・PPO/Adam更新・方策再推論0。

## 主要測定値

| 指令系列 | 物理KeyDown時刻（左指、秒） | 件数 | 4.327秒のKeyDown | 新規採点 |
|---|---|---:|---|---|
| B（更新前、元指令） | 3.718、4.019 | 2 | なし | 保存済みHit1/Extra1 |
| A→none（更新後、step362=none） | 3.718、4.019、**4.327** | 3 | あり | 保存済みHit1/Extra2 |
| **Bのstep413だけAに交換** | 3.718、4.019、**4.327** | **3** | **出現** | **未採点** |
| **A→noneのstep413だけBに交換** | 3.718、4.019 | **2** | **消失** | **未採点** |
| B陰性対照 | 3.718、4.019 | 2 | なし | 未採点 |
| A陰性対照 | 3.718、4.019、4.327 | 3 | あり | 未採点 |

- step413での指令はBがaction `0`（左指 `-1.0`）、A→noneがaction `2`（左指 `+1.0`）。各介入は**この1stepのみ**を変更。step453は各自の元指令を維持。
- D26で検出した有効左指指令差は**step413と453の2件のみ**。step453は追加KeyDownより後のため、今回の介入で変更していない。
- 介入による最初の1ms身体状態差は**4.131秒（tick index 4130）**。step413開始前の全1ms状態は各原本と完全一致。
- B→Aの単一指令介入はA→noneの全KeyDown列と一致し、逆介入はBの全KeyDown列と一致。**余分なKeyDownの件数差はそれぞれ+1、−1**。
- **6系列×4,800物理tick＝28,800物理tick、合計15物理KeyDown**。陰性対照の全1ms状態とイベントは完全一致。元B・A→noneのイベント列はD22原本と1e-8秒以内で一致。新規Hit/Extra採点0。
- 結果分類の成功は**事前登録した実験成立条件**であり、KeyDown反転を事後的な成功基準にしたものではない。

## 否定的結果と限界
- D19の非介入sampled評価は576目標に対してHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。今回も低速打鍵の学習改善なし。
- step413だけでこの保存済み1ケースの**4.327秒のKeyDown有無**を説明できるが、全てのExtra・Hit・長期学習失敗を説明したことにはならない。新しい介入系列は**未採点**で、保存済み採点を新規測定とみなさない。
- A0単指・1 seed・開ループ指令のみ。A1二指、A3未学習対照、他seed、閉ループの方策反応、教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002承認は未証明。
- 次の候補はstep413の直前の8次元観測・方策出力確率と報酬の対応を**保存済みD22記録から**調べること。ただし本D27での因果介入とは別の事前登録と独立Draft PRが必要。

## 実装・検証・記録
- 実装：`scripts/a002v2_l0_d27_step413_single_command.py`、専用11テスト、`.github/workflows/research-a-v2-l0-d27-step413-single-command.yml`。
- [初回成功 Actions #41556452950](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41556452950)：**11 passed**、原本取得・6系列再生・照合・JSON保存成功。Node.js 20非推奨警告のみ。
- [結果JSON・tests.log・run.log artifact #13193788540](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41556452950/artifacts/13193788540)（30日保持）。artifact ZIP SHA256：`a0c4e0dc64c541e0f502411acc67bd55b901d1d3012fb7baa7934e690864ff3c`。
- **Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
