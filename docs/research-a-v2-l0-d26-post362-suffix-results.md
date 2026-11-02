# 研究A v2-L0-D26 — 3.620秒後の指令列交換：測定結果

**判定：`valid_post362_suffix_swap`（SUCCESS）。A→noneに残った3番目の物理KeyDownは、元Bと元Aのstep363以降の実行指令列を交換すると、双方向で出現・消失した。** 教師なし学習改善の証拠ではない。

## 実行前固定と分離
- 実装・測定前の[事前登録コミット `c1156d41`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/c1156d4153d14c40f5906e112f1c69d92675dd8b)、[固定プロトコル](research-a-v2-l0-d26-post362-suffix-protocol.md)。結果を見て判定基準を変更していない。
- 独立[Draft PR #69](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/69)、baseはD25 Draft #68（`1f562bb3`）。監査PR #34・基準点PR #44・過去の研究PRは変更していない。
- D22成功[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995) の[artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296)から `result.json` と `steps.json` を原本照合。A0単指、seed320403、240ms間隔、開始1.075s、16目標、480制御step、10ms制御・1ms物理。保存済み指令のみ再生、方策再推論・新規学習・PPO/Adam更新は0。

## 実測：交換すると3番目のKeyDownが反転

| 指令列 | 物理KeyDown時刻（左指、秒） | 件数 | 元の保存済み採点 |
|---|---|---:|---|
| B（更新前） | 3.718、4.019 | **2** | Hit1、Extra1 |
| A→none（更新後・step362をnoneに変更） | 3.718、4.019、**4.327** | **3** | Hit1、Extra2 |
| **B→A-suffix**（step363以降をA→noneから交換） | 3.718、4.019、**4.327** | **3** | **未採点**。A→noneと全KeyDown列一致 |
| **A→B-suffix**（step363以降をBから交換） | 3.718、4.019 | **2** | **未採点**。Bと全KeyDown列一致 |
| B陰性対照 | 3.718、4.019 | 2 | 未採点 |
| A陰性対照 | 3.718、4.019、4.327 | 3 | 未採点 |

- **原本のBとA→noneのstep0〜362**：行動番号の違いは2step（最初はstep325）だが、物理的に有効な左指指令の違いは**0step**。両者のstep362はnone。
- **step363〜479**：行動番号の違いは**2step**、有効な左指指令の違いも**2step（413、453）**。追加KeyDownは4.327秒（source step432）なので、step453は時間的にその原因になれない。step413（4.130秒）が介入候補だが、**この実験ではstep413単独の効果は未検証**。
- 交換系列のstep362終了までの全1ms状態は対応する元系列と完全一致。陰性対照は全1ms状態とイベントが完全一致。BとA→noneの元系列の全KeyDownはD22原本と一致。
- **6系列、合計28,800物理tick、15物理KeyDown**。追加のHit/Extra採点0。元系列との物理イベント列の同一性のみを主張し、交換系列の新規Hit/Extraを計測したとは扱わない。

## 否定的結果と制限
- **step362の1回の指令だけで、後続の余分なKeyDownまで説明できる**という説明は否定された。後続の元の実行指令列が、少なくともこの保存済みA0の物理再生では追加KeyDownの有無を変えた。
- D25で確認した「介入後の方策反応による追加の選択行動差0件」と矛盾しない。D26は**PPO更新前後の元の方策同士の後続指令差**を調べたものであり、介入で新たに発生した差を調べたものではない。
- D19の非介入sampled 576目標でHit **5→4**、Extra **44→46**、KeyDown **49→50**。今回も低速打鍵の学習改善なし。A1二指・A3未学習対照・他seed・長期学習・人間同等性・単指物理的不可能性・正式研究002承認は未証明。
- 交換後は保存済み指令を開ループで再生している。閉ループ方策の再推論結果、学習改善、特定の1stepの唯一の因果効果には一般化できない。

## 検査・失敗の保存
- 最初の[Actions #41549105486 — FAILURE](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41549105486)：**8 passed、1 failed**。テスト用にafter側のstep100だけを変えたため、D25の原本介入前prefix同一性検査に先に失敗。**事前登録・実装・判定条件は変更せず**、テスト用のafter_to_none側にも同じstep100変更を加えて原本の構造を保つよう修正。[失敗ログartifact #13190805248](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41549105486/artifacts/13190805248)。修正コミット [`30c5c2e9`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/30c5c2e9f256bfa7738e7cc04a658c009d64f468)。
- [Actions #41549201051 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41549201051)：**9 passed**、原本取得・6系列物理再生・照合・結果JSON保存すべて成功。Node.js非推奨警告のみ。[結果JSON・tests.log・run.log artifact #13190755574](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41549201051/artifacts/13190755574)（30日保持）。

## 次の独立研究候補
**step413の1回の有効左指指令だけ**をB/A→noneで交換し、4.327秒の追加KeyDownが反転するか検証する。これはまだ未実施であり、別の事前登録とDraft PRが必要。

**Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
