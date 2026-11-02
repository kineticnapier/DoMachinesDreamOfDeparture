# 研究A v2-L0-D25 — 介入後の行動列固定と物理KeyDownの結果

**実行結果：`valid_frozen_suffix_comparison`（SUCCESS）。否定的発見：D22の介入後、step363〜479で実際に選択された行動は非介入の対応系列と1件も変わっていなかった。** D22で「方策の下流反応を含む」と留保していたが、この保存済み1ケースでは、少なくとも**実行指令列の変化**という経路は観測されなかった。確率分布や観測が不変だったという意味ではない。

## 事前登録・分離
- 実装より前に条件・比較・停止基準を固定：[事前登録コミット `005ac564`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/005ac564ff2e8b23052d485ee349204d35999c0c)／[プロトコル](research-a-v2-l0-d25-frozen-suffix-protocol.md)。実測後に変更なし。
- 独立Draft PR [#68](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/68)、baseはD24 Draft PR #67のHEAD `a2b11143`。監査PR #34、基準点PR #44、D22〜D24の成果物・モデル・報酬・PPO・Motor本体は変更していない。
- D22最終成功[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995) の[artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296)の `result.json` と `steps.json` を原本検査して使用。A0単指、240ms間隔、開始1.075s、seed320403、16目標、step362=3.620s、480制御step、1ms物理。

## 主要測定値

| 比較 | 非介入初回KeyDown | 介入・方策反応あり | 介入・後続指令固定 | step363〜479の選択行動差 | 物理的に有効な左指指令差 |
|---|---:|---:|---:|---:|---:|
| B→left | 3.718s | **3.689s** | **3.689s** | **0/117** | **0/117** |
| A→none | 3.689s | **3.718s** | **3.718s** | **0/117** | **0/117** |

- B→left：介入後の方策反応あり・後続指令固定のどちらも初回KeyDownがBより**29ms早い**。
- A→none：どちらもAより**29ms遅い**。
- 介入後の方策反応ありと後続指令固定では、**全物理KeyDownの時刻・指・順序・件数が完全一致**。B→leftは3件、A→noneは3件。
- **後続指令差0件**のため、初回KeyDown前の有効な左指指令差も0件。保存済みのD22介入系列の行動列は、各非介入系列からstep362の1指令を変えただけだった。介入後の観測・行動確率には差があり得るが、実際の抽選行動には差がなかった。
- D22でA→noneのExtraがBより1件多かったことを、**A→none介入後の追加の行動変化**に帰する説明はこのケースで否定的。元のAとBの行動列の差などは未分解。

## 実行・照合
- **8系列×480制御step×10物理tick＝38,400物理tick**、物理KeyDown合計**22件**。内訳はB2、A3、B→left3、A→none3、B後続固定3、A後続固定3、B陰性対照2、A陰性対照3。
- 保存済みD22の4系列について全物理KeyDownの時刻・指・件数が一致。B/A陰性対照は全1ms状態・イベント完全一致。4つの介入系列はstep362より前の全1ms状態が元系列と一致。固定後続の2系列はstep362以外の指令変更0件。
- 新規学習0、PPO更新0、Adam更新0、方策推論0、新規Hit/Extra採点0。**新しい系列は未採点**であり、過去のHit/Extraを機械的に流用していない。
- 専用テスト**11件成功**。初回[Actions #41541618509 — SUCCESS](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41541618509)。[結果JSON・3.620〜3.730sの1ms状態・検査/実行ログ artifact #13185922907](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41541618509/artifacts/13185922907)（30日保持）。Node.js 20非推奨警告が出たが実験は成功。失敗したActions実行はなし。

## 否定的結果・限界
- **方策反応が後続の選択行動を変え、29msの時刻差を生んだ**という説明は、この1ケースの保存済み行動列には当てはまらない。step362の1指令と、その後のMotor物理状態変化だけで、この比較における実行指令と物理KeyDown列の差が再現された。ただし**身体状態のみの唯一の因果経路**や他条件での一般化は未証明。
- D19の非介入sampled評価576目標ではHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。D25は新しい学習効果を測定しておらず、低速打鍵獲得の証拠はない。
- A1二指、A3未学習対照、他seed、長期学習、人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。左右使用率・交代率を成功条件にしていない。
- 次の独立検証候補は、A→noneに残った余分なKeyDownを、B/A間の**介入前後の元の行動列差**とMotor状態の差に分けて追跡すること。別の事前登録とDraft PRが必要。

**Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
