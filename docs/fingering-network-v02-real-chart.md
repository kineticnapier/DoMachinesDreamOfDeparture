# Fingering Network v0.2 — Real-chart curriculum pilot

## 判定

**FAIL（学習開始条件不足）**。Train-onlyの実譜面inventory、複雑区間抽出、
12軸・16接点物理教師データ生成、SHA256単位split、品質検査、HTML viewerは完成した。
しかし保存済みの信頼できる難易度は全80譜面がP11–P20で、G10以上は0譜面だった。
また、exact同時刻のmulti-contact eventも0件だった。このため人工データをG10相当と称して
補わず、v0.2の新規学習は実施していない。

難易度名はTUForumsの保存済みdataset manifestだけを使用した。TUForumsの公式
[API documentation](https://api.tuforums.com/docs/) とdifficulty catalogを参照したが、
実行時のMCP API probeは404だったため、現在値を推測・上書きしていない。

Qualification20、Reserve20、Finalは開封していない。U9「命日」とNerfed系列は今回の
Train80には存在しなかった。生成器は名前・manifest metadataで「命日」またはNerfedを
検出した場合、同系列の候補から除外する（unit testあり）。

## Phase 1: Train inventory

`data/DMDOD-v140-tuf-p11-p20/Train`の80譜面を全てcompileし、譜面byte列の
SHA256、manifest難易度、BPM、event密度、短間隔、special marker、full rotation、
決定論的な複雑区間を記録した。全行は
`reports/fingering-real-chart-v02-inventory-decd0f2.json`にある。

| 項目 | 結果 |
|---|---:|
| charts / unique SHA256 | 80 / 80 |
| difficulty | P11–P20、各8譜面 |
| G10以上を検証できた譜面 | 0 |
| total duration | 17,562.994s |
| events / targets | 114,259 / 114,259 |
| interval <=35ms / <=60ms | 16,595 / 25,937 |
| minimum interval | 0.012675ms |
| SetSpeed / Twirl / Pauseを含む譜面 | 78 / 79 / 37 |
| full rotationを含む譜面 | 68 |
| MultiPlanet / exact multi-contact event | 0 / 0 |
| selected candidate windows | 160（各譜面最大2、非重複） |

compiled BPMは5.305–35,839.999だった。極端な値とsub-10ms intervalは疑似BPM・
特殊構造を含む可能性があり、難易度ラベルとは扱わない。初回pilotではnear-zero intervalで
density ratioが発散したため、ratio寄与を上限9に固定し、`sub_control_interval`を独立した
反例カテゴリにした。

## Phase 2–3: bounded physical teacher pilot

固定budgetは4 charts × 1 window、focus最大16 events、直前prefix最大6 events。
各候補は同一のMotor stateから10ms control / 1ms physicsで最大3 eventをpreviewする。
採用候補だけを実環境へcommitし、候補ごとにscore、missing、extra、timing error、
不適格理由（`pressed` / `reserved`等）を保存した。exact AutoPlay時刻と候補物理結果は
教師・評価器側だけにあり、Network入力は現在のMotor状態、最大3件のTiming Oracle相対時刻、
chord size、候補物理軸allocationだけである。

split単位はchart SHA256。fit 3 charts / 44例、quality 1 chart / 18例で重複0。
重複時間窓はなく、失敗した教師decisionは学習例に保存しない。今回は62/62 focus eventが
実Hitだったため62例を保存した。同等good候補（best cost +1以内）が複数あるeventは30件。
候補理由の総数はphysical lookahead 489、pressed 87、reserved 44だった。

選択された譜面はP16/P17/P17/P19。うちCiforance MEGAMIX区間は最小7.813msで、
10ms controlより短い反例を明示的に1件含む。

## 同一条件の物理比較

全方式は同じ4 window、同じexact AutoPlay event時刻、同じ12軸・16接点Motorで評価した。
Networkは既存synthetic v0.1 checkpointのread-only評価であり、新規学習ではない。

| policy | Hit/Target | Miss | TooEarly | Overload | X accuracy | Timing MAE | mean / max candidate regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| CenterFirst | 87/87 | 0 | 0 | 0 | 96.724% | 14.881ms | 18.435 / 175.000 |
| Teacher v0.2 | 87/87 | 0 | 0 | 0 | 99.425% | 7.985ms | 0 / 0 |
| Network v0.1 | 87/87 | 0 | 0 | 0 | 99.713% | 8.529ms | 4.134 / 20.000 |

focus 62 eventsでのexact Teacher choice / equivalent-good率は、CenterFirstが
11.29% / 14.52%、Network v0.1が38.71% / 45.16%。TeacherとCenterFirstの採用指一致は
4/62（6.45%）で、TeacherがCenterFirst一辺倒になっていないことを確認した。

この小規模pilotでは既存NetworkがCenterFirstより低regret・低MAEだが、G10以上でも
multi-contact curriculumでもないため、v0.2学習の有効性を示す結果ではない。

## 可視化・生成物

- inventory: `reports/fingering-real-chart-v02-inventory-decd0f2.json`
- physical report: `reports/fingering-real-chart-v02-pilot-010fc19-r4.json`
- dataset: `artifacts/fingering_real_chart_v02/pilot-010fc19-r4.pt`
- viewer index: `artifacts/fingering_real_chart_v02/viewer-010fc19-r4/index.html`
- statuses: `logs/fingering-real-chart-v02-inventory-decd0f2.status.json`,
  `logs/fingering-real-chart-v02-pilot-010fc19-r4.status.json`
- stdout/stderr logs: matching `.log` files in `logs/`

各window viewerは12物理軸timeline、採用指、全候補、physical score、継続候補、
不採用理由、実KeyDown、Hit/Missを表示する。

最終dataset SHA256:
`0dea0d446738489dfaa8addd7483d420c923abe8bb5fd82957779d62aca55913`

inventory時間47.872s、最終physical pilot時間71.950s。Windows仮想環境ランチャーの
warningはstderrに残るがexit codeは0。先行runでcp932が「▷」をencodeできず失敗したため、
譜面名が物理runを中断しないbackslash-replacementを追加し、別write-once出力で再実行した。

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_fingering_real_chart_v02.py `
  tests/test_fingering_network_v01.py `
  tests/test_oracle_fingering_v02.py -q `
  --basetemp artifacts/pytest-fingering-real-v02-repro

.\.venv\Scripts\python.exe -m scripts.build_fingering_real_chart_v02 `
  --dataset data/DMDOD-v140-tuf-p11-p20 `
  --output reports/fingering-real-chart-v02-inventory-repro.json `
  --status-file logs/fingering-real-chart-v02-inventory-repro.status.json `
  --max-windows-per-chart 2 *> logs/fingering-real-chart-v02-inventory-repro.log

.\.venv\Scripts\python.exe -m scripts.generate_fingering_real_chart_v02 `
  --inventory reports/fingering-real-chart-v02-inventory-repro.json `
  --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt `
  --dataset artifacts/fingering_real_chart_v02/pilot-repro.pt `
  --report reports/fingering-real-chart-v02-pilot-repro.json `
  --viewer-dir artifacts/fingering_real_chart_v02/viewer-repro `
  --status-file logs/fingering-real-chart-v02-pilot-repro.status.json `
  --max-windows 4 --max-focus-events 16 --prefix-events 6 `
  *> logs/fingering-real-chart-v02-pilot-repro.log
```

## 次の課題

信頼できるG10以上のTrain譜面を、譜面系列・SHA256単位の新しいTrain-only sourceとして
追加する必要がある。特にexact同時刻multi-contact、MultiPlanet、指の復帰を要求する
連続chordが不足している。追加後は同じ固定budgetでinventoryと教師Hit/Missを再検証し、
quality gateが全てPASSした場合だけ、固定1回のv0.2予備学習へ進む。

Stage A FAIL SHA256は
`c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`、
MaleCNS4096 SHA256は
`a9d00a26c24ad19e815a0d768851f8040e00748e3a6474a31ce786576f53c710`で開始時から不変。
