# DMDOD HUD v3 Phase 5 — Future SetSpeed observability

## 結論

Train bool の 8.0x / 1.0x 反事実について、**44ms deadline 前に実画面または実音声から識別できることが確認済みの観測源はまだない**。したがって Phase 5 の総合判定は
`NO_CONFIRMED_OBSERVATION_SOURCE_BEFORE_DEADLINE` であり、新sensor・Timing・Motor統合には進まない。

ただし、Phase 4 の「同一の SetSpeed marker」という抽象化より強い stock-game 候補をDLLから発見した。`scnGame.ApplyEventsToFloors()` は SetSpeed の相対倍率を粗いカテゴリへ変換し、`scrFloor.UpdateIconSprite()` は通常play中に Rabbit / DoubleRabbit / Snail / DoubleSnail を描画する。固定反事実では 8.0x が `DoubleRabbit`、1.0x が非表示になるため、**Floor 1 のspriteがdeadline前の実フレームに写っていれば**二世界を識別できる。

これはまだ `OBSERVABLE_BEFORE_DEADLINE` ではない。保存済みbool動画は片側だけで、反事実との同条件pixel captureがなく、Floor 1 spriteの最初の可視frameとcapture latencyを測れていないため `UNVERIFIED` とする。さらにこのアイコンは数値表示ではなく多対一のカテゴリであり、未知の倍率8.0を一意に復元できない。

## 固定条件

| 項目 | 値 |
| --- | --- |
| chart | `04042 - bool - vo[ICE].adofai` |
| chart SHA256 | `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1` |
| inspected DLL SHA256 | `0d1524dabf0f8c67aa58ba1992d6ed858522b9094490ebbb00e5e8aeff7f40cd` |
| intervention | Floor 1 `bpmMultiplier` 8.0 -> 1.0; other chart fields unchanged |
| first event | 3.999333624s in both worlds |
| second event | 4.020163490s / 4.165972552s |
| earlier-world 44ms deadline | 3.976163490s |
| legacy result | 398 HUD278 frames through 3.970s equal; maximum difference 0.0 |

True due times, compiled floor speeds and intervention values above are evaluator-only. Studentへ渡してよいのはtimestamped pixel/audio detectionとその過去履歴だけである。

## Phase 5-A 観測候補

| 候補 | 実際の表示・音 | 最初に差が出る時刻 | 反事実識別 | 判定 |
| --- | --- | ---: | --- | --- |
| HUD278 / planet motion / countdown | Phase 4でモデル化済み。最初の到達までは両world同一 | 3.970sまで差なし | deadline前は不可 | `INDISTINGUISHABLE` |
| stock SetSpeed speed icon | DLL上は8x=`DoubleRabbit`、1x=`SameSpeed`; 通常playでは`SameSpeed` spriteは消える | renderer生成時点はplay前だが、実pixel onsetは未測定 | source semantics上は可能 | `UNVERIFIED` |
| stock generic event icon | `LevelEventType` iconはeditor pause側。通常playで未来数値を文字表示しない | 該当なし | 数値識別不可 | `INDISTINGUISHABLE` |
| JRP normal TBPM/CBPM/KPS | Show時は両方 `60.01 / 15 / 0.25` | 最初のHit hook 3.999333624s以後 | deadlineより23.170134ms遅い | `INDISTINGUISHABLE` |
| JRP Jongyeol pseudo | Show時は両方 `60.01 / 15 / 0.25`; startup 60.01 BPMではpseudo分岐なし | 最初のHit hook 3.999333624s以後 | deadlineより23.170134ms遅い | `INDISTINGUISHABLE` |
| ShowBPM等の他Mod | 公開READMEはTile/Real BPM/KPSを列挙するが、未来Floorの数値previewと更新時刻は確認できない | 未確認 | 未確認 | `UNVERIFIED` |
| audio | chartは両worldとも`bool.ogg`; countdown設定も同一 | 対になる実録音なし | asset/capture不在 | `UNVERIFIED` |

### Stock icon の根拠と限界

同一DLL buildの逆コンパイルとIL indexを確認した。

1. `ApplyEventsToFloors` はSetSpeed後の`floor.speed`と前floorのspeedの相対差を計算する。
2. 5%以下は`SameSpeed`、増速はおおむね2.05x未満が`Rabbit`、それ以上が`DoubleRabbit`、減速は`Snail`/`DoubleSnail`となる。
3. `UpdateIconSprite` はRabbit/Snail系をgameworldでも描画する一方、`SameSpeed`はcontroller pause中だけ描画する。
4. generic `eventIcon` spriteはeditor pause側の表示で、通常playの未来数値previewではない。

したがって今回の8x/1x pairはrenderer内部では異なる。しかし、renderer enumや`floor.speed`をsensorへ直接渡すことは禁止する。screen cropに実際に現れたspriteだけを、validity、capture時刻、latency付きで観測して初めてpolicy-safeとなる。

また`DoubleRabbit`は倍率8.0の符号化ではない。同じカテゴリに複数倍率が入るため、アイコンだけから第2eventの正確なdueを計算してはならない。確認後のTiming-only実験でも、カテゴリから保証できるdue区間が44ms準備に十分な場合だけconfirmedとし、それ以外は棄権する。

### JRP

固定upstream `Jongye0l/JipperResourcePack@861a1ff6` のnormal modeは、表示開始時とHit hookで`UpdateBpm`を行う。式はcurrent floorのTBPMとcurrent-to-next floor間隔のCBPMであり、未来SetSpeed数値の専用previewではない。

固定pairではstartup floor 0からfirst eventまでの間隔が同一なので、normal/Jongyeolとも表示開始時は完全一致した。Floor 1へ到達した後の表示は異なるが、最も早い理想的なHit hookでも3.999333624sで、deadline後である。実pixel refreshはこれより早くならない。

上流ソース:

- [JRP Overlay.UpdateBpm](https://github.com/Jongye0l/JipperResourcePack/blob/861a1ff6d12ef1c785ca16fb980a7e9292cfee4f/JipperResourcePack/OverlayContents/Overlay.cs#L614-L638)
- [JRP repository feature list](https://github.com/Jongye0l/JipperResourcePack)
- [ShowBPM feature list](https://github.com/FLOWERs-Modding/ADOFAI_ShowBPM)

### Audio

Phase 4 interventionは`bpmMultiplier`だけを変更し、`songFilename`、countdownTicks、first dueを変えない。従って同じaudio assetを使う設計上、deadline前のmusic/countdown audioが未来の任意SetSpeed数値を符号化する根拠はない。しかしTrain80配布物には`bool.ogg`がなく、反事実pairのloopback recordingもない。波形prefix equalityとcapture latencyを実測していないため、ここでは`INDISTINGUISHABLE`へ格上げせず`UNVERIFIED`とした。

音楽の拍から作者の意図を推測することはできても、任意のchart SetSpeed値を一意に復元する因果的センサーにはならない。同一音源に複数の合法なSetSpeed値を与えられるためである。

## Phase 5-B 監査fixture

`scripts/audit_setspeed_observability_v01.py` は次を分離して記録する。

- renderer/evaluator: compiled pair、stock DLL icon分類、posthoc due/deadline
- sensor evidence: 実pixel/audio captureの有無、observed time、latency
- policy boundary: timestamped visible/audible observationだけを許可

実captureがない状態でrenderer予測を観測に偽装せず、stock iconを`UNVERIFIED`に保つ。JRPについては表示文字だけを比較し、最初の差がdeadline後であることを確認する。出力は`reports/hud-v3-phase5-setspeed-observability-20261019-r1.json`。

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_audit_setspeed_observability_v01.py --basetemp artifacts/pytest-phase5-setspeed-r2
.\.venv\Scripts\python.exe scripts/audit_setspeed_observability_v01.py --output reports/hud-v3-phase5-setspeed-observability-YYYYMMDD-rN.json
~~~

スクリプトは既存outputを上書きしない。

## Phase 6判断

sensor実装の前に、次の1回の固定capture試験だけを行う。

1. original 8.0xとcounterfactual 1.0xを同一game build、同一camera/Mod設定、同一audioで録画する。
2. Floor 1を含むcropと音声を保存し、実表示frame時刻とcapture latencyを記録する。
3. `DoubleRabbit`対非表示が最初に検出された**観測到着時刻**を求め、3.976163490s以前か判定する。
4. deadline以前なら、このpairに限り`OBSERVABLE_BEFORE_DEADLINE`とし、versioned icon sensorを設計する。値はicon class + valid + observed_at + latencyだけとし、倍率やfloor IDは含めない。
5. 見えない、遅い、設定依存で再現不能なら`UNVERIFIED`または`INDISTINGUISHABLE`を維持する。
6. どちらの場合も、カテゴリから倍率8.0を直接復元しない。Timing-onlyでdue区間とabstentionを評価してからでなければMotorへ接続しない。

現段階ではHUD278のPhase 4結論を維持する。stock speed iconは有望な追加センサー候補だが、9/9達成済みとは扱わない。
