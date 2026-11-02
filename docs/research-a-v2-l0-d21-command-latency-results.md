# 研究A v2-L0-D21 — 行動指令と物理KeyDownの遅延（実測結果）

**判定：`valid_single_case_command_latency`。学習効果は否定的。** 実装前の事前登録 [`7a2bbf22`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/7a2bbf2273462598112ec7f512510df308a2c006) の条件・停止・評価基準は変更していない。新規学習0、PPO/Adam更新0。D19の1試行を更新前・更新後・更新前陰性対照の計3回だけMotorで再評価した。

## 実行と保存
- 独立Draft PR #64、base D20 PR #63の `research/a-v2-l0-d20-archived-first-keydown-divergence`。既存の基準点PR #44、監査PR #34、D19 #62、D20 #63は変更していない。
- [GitHub Actions #41504659145](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41504659145) SUCCESS（最初の実装HEAD）。[Actions #41504686737](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41504686737) SUCCESS（入力の厳密照合を事前登録どおりの物理trace・行動計数・step数に限定した後のHEAD）。専用 **8テスト成功、NumPy任意依存警告1件**、Actions側Node.js非推奨警告。実験そのものは正常終了。
- [後者の結果artifact #13169373365](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41504686737/artifacts/13169373365)：`result.json`、`steps.json`（3試行の全480制御stepと各物理KeyDownの対応）、`tests.log`、`run.log`。GitHub Actionsで30日保持。結果の要点は本書に永続保存。
- 実行環境：Python 3.11.17、torch 2.5.1+cpu、1スレッド、native CPU経路（AMD EPYC 7763 / AVX2）。D8初期重み、D12 native更新後重み、凍結バッチのSHA照合成功。D19保存済み全45対の検証も成功し、対象1対の物理trace、Hit/Extra、行動計数、480 stepが**完全一致**。更新前の陰性対照は全step履歴も完全一致。
- 最終結果ファイルSHA256：`result.json` = `158011b205f9521b7868176dcbfbbc5285e97a370b7e13b39f9b4ce083446dbf`、`steps.json` = `a361968514ee48af320c1b99de5ff767a4158fe3a5b8766eb1915014641b216c`。

## 単一試行の実測（A0単指、sampled、case4/repetition3）
間隔240ms、開始1.075秒、seed320403、16目標。右指はA0で物理的に無効。4行動の符号は none=0 / right=1 / left=2 / both=3。

| 指標 | 初回PPO更新前 | 更新後 |
|---|---:|---:|
| Hit | **1** | **0** |
| Extra | **1** | **3** |
| 実際の物理KeyDown | **2** | **3** |
| 制御step | 480 | 480 |
| 最初の物理KeyDown | **3.718s** / left / Hit | **3.689s** / left / Extra |
| 目標11への偏差 | **+3ms** | **−26ms** |
| 最初のKeyDownを含む制御step | 371（3.710s開始） | 368（3.680s開始） |
| 最初のKeyDown直前の左指押下指令開始 | **3.650s** | **3.650s** |
| その指令開始からKeyDownまで | **68ms** | **39ms** |

**重要：最後の押下指令開始時刻は同じ3.650sだったが、物理KeyDownは29msずれた。** この遅延差を、身体固有の固定遅延が29ms変わった証拠とは扱わない。直前の身体状態が異なるためである。

## 最初の分岐を時系列で確認
- **3.250s（step325）**：最初の**行動番号**相違。更新前はnone(0)、更新後はright(1)。A0では右指が無効なので、この2指令は**物理的には同じ左指-1の指令**。この相違を物理的分岐と誤認しない。
- **3.500s（step350）**：更新前both(3)、更新後left(2)。これもA0では同じ左指+1の指令。
- **3.620s（step362）**：最初の**有効な左指指令**の相違。更新前none(0)→左指-1、更新後left(2)→左指+1。直前8次元観測はまだ同じ。更新後のみ3.620sに左指押下指令を再開。
- **3.630s（step363）**：最初の8次元観測の相違。更新前の左位置×500は **−0.19448639**、左速度×10は **0.14027337**。更新後はそれぞれ **−0.16183486**、**0.29792692**。この時点で身体状態の差が観測できる。
- **3.650s（step365）**：両方が左指+1へ切り替え、最後の左指押下指令開始時刻が一致。**3.620sの早い指令と、その後の身体状態の履歴は一致していない**。
- **3.689s（step368）**：更新後の最初の物理KeyDown、目標11より26ms早くExtra（即時報酬0）。
- **3.718s（step371）**：更新前の最初の物理KeyDown、目標11より3ms遅くHit（即時報酬0.9775）。

全480stepで左指の有効指令が異なったのは **3step**（3.620s、4.130s、4.530s）。4行動番号の相違は5step（3.250s、3.500s、3.620s、4.130s、4.530s）。更新前の行動計数 [151,124,125,80]、更新後 [147,125,129,79]。

更新前の2回目KeyDownは4.019s・Extra、更新後は4.003s・Extraと4.324s・Extraが追加。これによりこの1対の差はHit −1、Extra +2、KeyDown +1。D19全36 sampled対の集計も **Hit 5→4、Extra 44→46、物理KeyDown 49→50** で改善しない。argmaxの既存結果は両側0物理KeyDown。

## 否定的結果と解釈の制限
- **低速打鍵の学習成功は確認できない**。今回の1対はむしろHitが減りExtraが増えた。更新後に押下指令の時刻や身体状態が変わっても、Hit率改善の証明にはならない。
- 最初の行動番号差（3.250s）はA0の物理指令差ではなかった。実際の左指指令差は3.620sであり、最初の物理KeyDown差より前に発生した。ただし、3.620sの指令だけが29msの差を**引き起こした**とは未証明。次に調べるならこの1stepの指令を固定する介入を**別途事前登録**し、その他の条件を維持する必要がある。
- `last_left_command_onset_s` は直近の左指 -1→+1 の切替時刻を指す記述的な指標。Motorの内部積分状態、押し直し、過去の指令履歴の因果効果をこれ1値で表現できない。
- A1二指、A3未学習対照、身体介入は今回実施していない。教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。左右交代率・使用率は成功条件にしていない。
- 元の身体、報酬、PPO、重み、既存PRは変更していない。**Draft維持・自動マージ禁止・Draft解除禁止・マージ禁止。**
