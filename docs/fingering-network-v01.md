# DMDOD — Fingering Network v0.1

## 結論

Fingering Network v0.1 の固定小規模学習を1回だけ実行し、未知の固定BPM fixtureで
CenterFirst / Fingering Teacher v0.2 / Networkを同一AutoPlay時刻・同一12軸16接点Motorで
比較した。v0.1は48/48 Hit、Miss 0、TooEarly 0、Overload 0を維持しながら、CenterFirstの
Timing MAE 18.071msを10.126msへ改善した。Teacherと異なる物理部位を35/48イベントで
選び、その35イベントすべてをHitした。

固定成功条件はすべて満たしたため、**CenterFirstに対するv0.1実験はPASS**とする。
ただしTeacher v0.2のTiming MAE 8.591msには1.535ms届かず、validation系列の
equivalent-good rateも53.3%に留まる。Teacherを超えた、一般的な未知譜面に汎化した、
または実人体の運指を獲得したとは結論しない。

## Phase 1: Teacher v0.2確認

基準commit `fba25aafa629c32241271b06185aa64d953683c2`へfast-forward後、
`tests/test_oracle_fingering_v02.py`を実行した。sandbox既定TEMPの作成拒否を避けるため
workspace内の専用`--basetemp`を使用し、5/5 testsが成功した。

確認した教師構造:

- 12物理軸、16論理接点
- single / thumb tip / thumb flat / hand-side edge allocation
- 現在Motorをdeep-copyし、10ms command / 1ms physicsで最大3イベントをshadow評価
- 候補ごとにmissing、extra、KeyDown timing error、best continuation、scoreを生成
- shadow stateはlive Motorを変更せず、採用するのは先頭イベントallocationだけ

## Phase 2: データ

入力は次に限定した。

- 最大3イベントの現在からの相対AutoPlay時刻、chord size、presence
- 現在の12軸position、velocity、pressed state
- 各物理軸のavailability
- 現在のthumb modeとhand-side contact状態
- 評価対象候補の物理軸・接触mode

Teacher score、Teacher採用結果、shadow未来物理結果、論理keyboard binding名は入力に
含めていない。これらはlabelまたはposthoc evaluatorだけが使用する。

splitは隣接frameではなくpattern-series SHA256単位で固定した。

| split | series | examples | 用途 |
|---|---:|---:|---|
| Train | 6 | 56 | 80 epochの固定学習のみ |
| Validation | 2 | 15 | 報告のみ。model/epoch/threshold選択には未使用 |
| Test | 2 | 19 | 最終ranking報告のみ。選択には未使用 |

Trainにはalternating、rolling、short repeat、22ms high density、accelerando、2-contact
allocationを含めた。Validation/Testのseries SHA256はTrainと重複しない。
Qualification20、Reserve20、Finalは開封していない。

## Phase 3: Network

モデルはcontextと候補allocationを連結して1候補ずつscoreする共有MLPである。

- hidden width: 96
- parameters are trained with AdamW, learning rate `5e-4`
- fixed seed: 4101
- fixed epochs: 80
- training runs: 1
- hyperparameter search: none
- invalid / physically unavailable candidates are inference時にmask
- teacher cost差が1.0以下の候補はequivalent-good setとし、不要な二値化をしない
- listwise good-set loss、明確なcost差へのpairwise ranking、同等候補のtie regularizerを使用

Ranking結果:

| split | exact Teacher choice | equivalent-good | mean cost regret | max regret |
|---|---:|---:|---:|---:|
| Train | 55.36% | 96.43% | 0.393 | 8.0 |
| Validation | 46.67% | 53.33% | 2.800 | 15.0 |
| Test | 47.37% | 73.68% | 2.263 | 19.0 |

exact一致が低くてもequivalent-goodが高い例があり、複数の物理的に同等な解を許容する
目的は満たしている。一方validationの落差はseries汎化がまだ弱いことを示す。

## Phase 4: 物理比較

学習系列とは別の900/1600/2800 BPM ordinary-floor fixtureを用い、各16イベント、合計48
イベントをAutoPlay Timing Oracleで評価した。3方式は同じchart SHA、10ms control、
1ms physics、同じMotor profileを使用した。

| policy | Hit/Miss | TooEarly | Overload | KeyDown | X accuracy | Timing MAE |
|---|---:|---:|---:|---:|---:|---:|
| CenterFirst | 48/0 | 0 | 0/3 charts | 48 | 90.208% | 18.071ms |
| Teacher v0.2 | 48/0 | 0 | 0/3 charts | 48 | 97.708% | 8.591ms |
| Network v0.1 | 48/0 | 0 | 0/3 charts | 48 | 98.438% | 10.126ms |

BPM別Timing MAE:

| BPM | CenterFirst | Teacher v0.2 | Network v0.1 | Network X |
|---:|---:|---:|---:|---:|
| 900 | 16.312ms | 3.667ms | 6.458ms | 100.000% |
| 1600 | 20.250ms | 8.625ms | 10.000ms | 100.000% |
| 2800 | 17.652ms | 13.482ms | 13.920ms | 95.312% |

成功判定はユーザー指定の物理比較要件から次のように構成し、再学習や再評価をせず
保存済み1-runへ一度だけ適用した。これは実験前登録された統計基準ではないため、PASSは
この固定fixtureに対するengineering milestoneとして扱う。

- Network hit rateがCenterFirst未満でない
- Miss、TooEarly、OverloadがCenterFirstより悪化しない
- Network Timing MAEがCenterFirstより小さい
- Teacherと異なるallocationでも少なくとも1イベントをHitする

全条件がtrueだった。NetworkはTeacherと35件異なり、35件すべてHitしたため、単なる
Teacher label一致率だけによる成功ではない。

## Phase 5: Viewer

最終HTMLは12物理軸timelineを維持し、論理keyboard名の代わりに「左手・指1」
「右手・親指」「左手・側面」のような物理部位を表示する。各イベントで次を確認できる。

- Network allocationとTeacher allocation
- 全候補のNetwork score、Teacher score、eligible/reason
- 採用物理軸に対応付けた最寄り実KeyDown時刻
- Hit/Miss/TooEarly judgement
- Teacherと異なるイベントの強調表示

高密度時に隣接イベントのKeyDownを誤って併記しないよう、最終viewerでは採用軸ごとの
最寄りKeyDownをposthocに1件だけ対応付けた。この補正は学習・policy・物理集計を変更しない。

## 因果性と分離

- AutoPlay exact timesはNetworkへ相対future-event timingとして与えた。これは今回許可された
  Timing Oracleであり、Timing Studentは使用していない。
- Network推論はteacher rank、teacher cost、candidate shadow rolloutを呼び出さない。
- Networkがallocationを決定した後、viewer/evaluatorだけが同じstateでTeacher候補を評価する。
- invalid candidate maskは現在Motor stateとallocation topologyだけから計算する。
- evaluationのHit/Miss/X/Timing MAEは実際に継続したMotor/real-chart scorerから取得する。

## 保護確認

- Stage A report SHA256:
  `c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`
- MaleCNS4096 SHA256:
  `a9d00a26c24ad19e815a0d768851f8040e00748e3a6474a31ce786576f53c710`
- Stage A gateは迂回せず、既存checkpointを変更していない
- ユーザーの未コミット`configs/training/prestige1.toml`差分は変更・commitしていない

## 再実行

CPUの固定1-run:

```powershell
& .venv/Scripts/python.exe scripts/train_fingering_network_v01.py `
  --checkpoint checkpoints/fingering/fingering_network_v01-reproduction.pt `
  --dataset artifacts/fingering_network_v01/dataset-reproduction.pt `
  --report reports/fingering-network-v01-reproduction.json `
  --viewer reports/fingering-network-v01-reproduction.html `
  --status-file logs/fingering-network-v01-reproduction.status.json
```

関連テスト:

```powershell
& .venv/Scripts/python.exe -m pytest `
  tests/test_fingering_network_v01.py `
  tests/test_train_fingering_network_v01.py `
  tests/test_oracle_fingering_v02.py -q `
  --basetemp .pytest-tmp-fingering-v01-reproduction
```

## 成果物

- checkpoint: `checkpoints/fingering/fingering_network_v01-4fa2ae1.pt`
  - SHA256 `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
- dataset: `artifacts/fingering_network_v01/dataset-4fa2ae1.pt`
  - SHA256 `df0a8b0d07c8433652d51bbd5c6c0848c2ab3b90a95023117e4738d5144e4f49`
- raw report: `reports/fingering-network-v01-4fa2ae1.json`
  - SHA256 `d54b7f7e44f793ff0a56705de41ac1822ee56c266bbec14f7a41661b13b6813c`
- final analysis: `reports/fingering-network-v01-4fa2ae1-analysis.json`
- final viewer: `reports/fingering-network-v01-4fa2ae1-final.html`
- run status/logs: `logs/fingering-network-v01-4fa2ae1.*`

## 次の改善候補

今回はv0.1で停止する。次へ進む場合の優先候補は、validation seriesのcost regret反例を
family別に調べること、candidate scoreのscale calibration、現在stateに加えて短いMotor
historyを加えるablation、2-contact fixtureの実chart scorer改善である。大規模RLやHuman
Style学習へ進む根拠はまだない。
