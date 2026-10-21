# Phase 20B — Stock Floor State Machine & Scorer Parity

## 判定

`STOCK_MODEL_BACKLOG_REPRODUCED`

ただし、この判定は監査済みDLLの **NoFail強制遷移分岐** と、Phase 16で
継続実行されたrepository Scorerとの限定比較を意味する。通常Fail分岐では
同じ継続バックログにならず、Event 36で停止する。監査DLLと現在の実機buildの
一致、およびUnity frame schedulingを含む実機parityは未検証である。

## 証拠範囲

- 逆コンパイル資料: `F:\adofai-code`
- Assembly-CSharp.dll SHA256（Phase 19監査値）:
  `0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`
- `scrPlayer.cs` SHA256:
  `21326189390DB431A4BBE2EA9E608D67048509AA2138526E5AD5179DA08EFC47`
- `scrPlanet.cs` SHA256:
  `2250FEB39218F386233227E1BD2007BD1A9360C62C12749EBE0B8EFC07618C0E`
- `scrMisc.cs` SHA256:
  `5DD7696235ED26476A01D9417ACEE4A7BE91D19567A9BD83FA510DC4225F62A5`
- Phase 16 source:
  `reports/phase16-firsttown12-event89-causal-20261021-r2.json`
- First town SHA256:
  `4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1`

Phase 20Bの参照モデルはchart target timeを惑星角度の事後再構築に使用する。
これはEvaluator専用で、Timing/Fingering/Motorへの入力ではない。repositoryの
`target_index`はStock側の床選択には一切使わず、比較後にだけ照合した。

## DLL由来の通常床制御フロー

### KeyDownと床遷移

1. `scrPlayer.Simulated_PlayerControl_Update` (`scrPlayer.cs:401-440`) は
   `CheckPostHoldFail`を先に実行し、その後で入力queueを処理する。
2. `HitAutoFloors` (`:505-523`) は観測した各pressを`keyTimes`へ保存する。
3. `UpdateHoldKeys` (`:693-739`) はqueueを1件取り出して`Hit()`を呼ぶ。
4. `Hit` (`:741-830`) は`chosenPlanet.SwitchChosen()`を呼ぶ。
5. `scrPlanet.SwitchChosen` (`scrPlanet.cs:630-817`) は現在惑星の
   `targetExitAngle`だけを判定し、有効判定なら`currfloor.nextfloor`を選ぶ。
   別の未処理床を探索しない。
6. `scrMisc.IsValidHit` (`scrMisc.cs:269-292`) の通常設定では
   VeryEarlyからVeryLateだけが有効である。TooEarly/TooLateは
   `OnDamage`後も同じ床に留まる。
7. 有効時は`MoveToNextFloor` (`scrPlanet.cs:950-1177`) がcurrent floor、
   target angle、speed、directionを更新する。

`AsyncInputUtils.WhileFloorNotChange`はfloorが変わった場合に処理を再実行する。
同一frame内の複数KeyDownは1件へ畳まず、queue順に次のcurrent/next-floorへ
作用する。Midspinでは`Hit()`が内部queueを追加し (`scrPlayer.cs:820-827`)、
通常の2惑星Midspin bridgeを追加の物理KeyDownなしで通過する。この限定経路は
参照モデルで明示的にcollapseしている。

### 自動MissとFail/NoFail

`CheckPostHoldFail` (`scrPlayer.cs:453-475`) の判定はrepositoryの
`due + pass_s`ではない。

- 通常Fail: passed angleが
  `max(180°, 2 * CountedBoundary)`を **厳密に超える** と`Die()`。
- NoFail/safe/cooldown: thresholdは`CountedBoundary * 1.01`。
- 通常Failの`Die`は床を進めずplayを停止する (`:324-385`)。
- NoFail等では`noFailInfiniteMargin=true`で`Hit()`を呼び、
  `SwitchChosen`が判定を`FailMiss`にして次床へ強制遷移する
  (`scrPlanet.cs:639-652`)。

無効入力はfail barへ0.5 damageを加える。`scrFailBar.DidFail`はcounterが
1.0を **超えた** 場合だけ成立するため、frame decayの有無にかかわらず最初の
2回まではoverload deathにならない。3回目以降の正確なfail時刻はUnity Update
cadenceとdecayが必要なので、参照モデルは`NOT_COMPARABLE`へ降格する。

## 参照モデル

`dmdod.adofai.stock_floor_state_machine`はproduction Scorerから独立している。

- `StockFloorTarget`: next floor、due、approach BPM、pitch、margin scale、
  自動通過するMidspinを保持するEvaluator型。
- `StockFloorStateMachine`: current floor、next target、fail状態を保持する。
- 各KeyDownについて入力前床、対象床、Phase 20A判定カテゴリ、入力後床、
  transition、auto Miss、DLL flow、再現可能性を記録する。
- `FAIL`と`NO_FAIL`を明示指定し、暗黙に混合しない。
- 未対応特殊分岐は`UNSUPPORTED`で停止し、通常床として推測しない。

SetSpeedはdestinationに到着した後にspeedを更新するため、そのdestinationを
叩く判定窓には直前のapproach floor BPMを使う。途中`angleOffset`付きSetSpeed、
Pause、Hold、MultiPlanet等で一定角速度を保証できない場合は棄権する。

## Phase 16 A/B

固定区間は3.15–5.36秒、54 KeyDownである。比較対象は同じ保存済み物理入力列。

### NoFail分岐

| 指標 | 結果 |
|---|---:|
| KeyDown比較数 | 54 |
| Judged target一致 | 54/54 |
| 判定カテゴリ一致 | 54/54 |
| policy intended IDとのずれ | 50 |
| 最初のずれ | ID 38 → Event 37, 3.261 s, VeryLate |
| 最後のずれ | ID 89 → Event 88, 5.253 s, VeryLate |
| 再同期 | ID 90 → Event 90, 5.319 s, Perfect |

3.261秒の更新開始時、Event 36は314.880°遅れていた。1360 BPM Normalの
Counted境界は265.2°、NoFail thresholdは267.852°なので、入力処理より先に
Event 36がFailMissとしてfloor 39へ強制遷移する。その直後、ID 38のKeyDownは
current/next-floor規則でEvent 37 (floor 40)に作用しVeryLateとなる。

以後も各入力は一つ前のeventを有効判定として進める。5.253秒のID 89は
Event 88に消費される。5.319秒ではEvent 89が281.520°遅れでNoFail thresholdを
超え、まずFailMissとして強制遷移し、その後ID 90がEvent 90をPerfect判定して
再同期する。したがって、Phase 16の50 KeyDownバックログは監査DLLのNoFail
状態機械でも再現された。

比較区間のStock自動FailMissはEvent 36、41、46、89で、保存済みrepository
Scorerのexpiryと最終的な対象・カテゴリが一致した。ただしthreshold自体は
65msと65.65msで同一ではなく、この入力列で結果が一致しただけである。

### 通常Fail分岐

Event 35着地後をEvaluator snapshotとして固定し、次targetをEvent 36として
同じ入力を与えた。このsnapshotは分岐を切り分けるための事後初期化であり、
曲頭から通常Fail Stockがそこまで生存したという主張ではない。

| 時刻 | 入力 | 対象 | 結果 | 床遷移 |
|---:|---|---:|---|---|
| 3.261 s | ID 38相当 | Event 36 | TooLate | なし |
| 3.311 s | 次KeyDown | Event 36 | TooLate | なし |
| 3.355 s | 入力前CheckPostHoldFail | Event 36 | Fail/death | なし |

3.261秒がrepositoryとの最初の状態不一致点である。通常Fail thresholdは
`2 * 265.2° = 530.4°`であり、3.261秒の314.880°では自動Miss前、しかし
有効窓外なのでEvent 36へのTooLateとなる。3.355秒には698.400°となり、
入力処理前の`CheckPostHoldFail`で停止する。したがって通常Failでは
Event 37〜89の継続バックログを比較できない。

## 結論の責任範囲

- **判定カテゴリ**: Phase 20A strict-boundary modelを使用し、NoFail固定区間で
  54/54一致。
- **床状態**: NoFail固定区間で54/54対象一致。通常Failは3.261秒から分岐し、
  3.355秒に停止。
- **自動Miss**: NoFailの強制Hit/FailMissと通常Failの停止を別に再現。
- **Phase 16原因**: 継続Scorerに対応するNoFail条件では、Event 36の自動Miss、
  ID 38→Event 37消費、50入力のずれ、Event 89自動Miss後の再同期まで成立。
- **実機Stock**: DLL snapshotと実機buildの同一性、NoFail設定、実frame順序を
  実測していないため未検証。

## 未再現分岐

- Hold、strictHolds、requireHolding
- MultiPlanet、multipress、OverPress、multitap
- Pause中およびmid-floor SetSpeed angleOffsetの非一様角速度
- Checkpoint、safe floor、miss cooldown、lives、official-level終端例外
- PurePerfectOnly/PerfectsOnly、mobile、speed trialの状態遷移統合
- fail-barのUnity frame単位decayと3回目以降のinvalid-input death時刻
- auto floor、freeroam、co-op、drum controller、revive
- 現行実機buildでの同一性とrender/input frame ordering

これらへ遭遇した場合、モデルはStock通常床処理を推測適用せず
`UNSUPPORTED`または`NOT_COMPARABLE`を返す。

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_stock_judgement_reference.py `
  tests/test_stock_floor_state_machine.py `
  tests/test_audit_phase20b_stock_floor_state_v01.py

.\.venv\Scripts\python.exe scripts/audit_phase20b_stock_floor_state_v01.py `
  --output reports/phase20b-stock-floor-state-20261022-r1.json
```

監査JSONは既存ファイルを上書きしないよう`open("x")`で生成する。
