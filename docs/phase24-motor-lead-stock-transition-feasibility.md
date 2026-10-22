# Phase 24 — Motor Lead & Stock Transition Feasibility

## 結論

判定は次のとおり。

- `LEAD_44MS_JUSTIFIED`
- `SHORTER_LEAD_TEMPORALLY_FEASIBLE`
- `SHORTER_LEAD_UNSAFE`
- `NOFAIL_DELAYED_HIT_CONDITIONALLY_VALID`
- `STOCK_RUNTIME_PARITY_UNVERIFIED`
- `NO_SAFE_INTERVENTION_ESTABLISHED`

ここで`LEAD_44MS_JUSTIFIED`は、現在のシミュレータ用の**名目較正値として由来を確認できた**という意味である。全軸・全状態の理論下界、最悪位相保証、実機Motorの保証という意味ではない。productionの44ms、Timing、Fingering、Motor、Scorer、HUDは変更していない。

## A. 44msの由来

`calibrate_n_key_press_lead()`は、静止状態から最大正方向指令を与え、最初の物理KeyDownまでを1ms physicsで測定する。その後、理想的な指令時刻から次の10ms control decisionまでの**平均**遅延として半control tickを加える。

| 構成要素 | 値 | 性質 | 根拠と限界 |
|---|---:|---|---|
| inner pair cold press | 39ms | シミュレータ実測 | 12軸を今回再較正。`left_1/right_1`は39ms |
| control tick | 10ms | 固定契約 | policy commandは10msごと。半tick 5msは平均補償であり最悪10ms保証ではない |
| nominal lead | 44ms | 経験的較正 | 39+5ms。`NORMAL_LEAD_S=.044`として凍結 |
| physics tick | 1ms | 数値積分刻み | latency測定の分解能。leadへさらに1msを加える式ではない |
| 12軸 cold range | 39–43ms | シミュレータ実測 | `left/right_2,5=41ms`、`left/right_3,4=43ms`。44msは全軸の「latency+半tick」ではない |
| activation補償 | 0–25ms | 状態依存の追加Lead | 負activationだけに適用。軸固有cold latencyや全競合の一般保証ではない |
| reservation | dueの69ms以内で選択、due−44ms−extraでarm | scheduler契約 | unarmedは取消可能。armedは物理brakeが必要で即時取消ではない |
| pressed/release | KeyDown後は実KeyUpとreset位置まで占有 | 物理状態契約 | elapsed/canceledだけでは軸を解放しない |
| 同時・連打 | 空き軸、hand capacity、contact topologyに依存 | 状態依存 | 44ms単独では競合なしを証明しない |
| Stock判定窓 | Motor leadとは独立 | DLL参照モデル | 遅いKeyDownがcounted window内でも、due alignedとは限らない |

したがって44msは「center-first inner pairのcold 39ms＋平均量子化5ms」を起源とする名目値である。保守的成分は5msあるが、最悪control位相、43ms outer axis、予約競合を同時に包む上界ではない。既存の実譜面物理評価が契約を支える一方、数式だけから普遍的安全性は導けない。

## B. Event 37 Lead感度

事後truthはdue `3.220588272s`、Stock counted windowは`[3.155588272, 3.285588272]s`。現行point forecastは3.190sに初めて得られ、HUDとして確認済みである。表のKeyDownは、Event 37が実際には未割当だったため、**空いている最遅cold軸43ms**を仮定した条件付き投影である。

| Lead | 名目期限に間に合う | 最初のcommand tick | cold KeyDown | due誤差 | dueまでに押下 | Stock窓内 |
|---:|:---:|---:|---:|---:|:---:|:---:|
| 44ms | No | 3.190s | 3.233s | +12.412ms | No | Yes |
| 30ms | Yes | 3.200s | 3.243s | +22.412ms | No | Yes |
| 20ms | Yes | 3.210s | 3.253s | +32.412ms | No | Yes |
| 10ms | Yes | 3.220s | 3.263s | +42.412ms | No | Yes |
| 0ms | Yes | 3.230s | 3.273s | +52.412ms | No | Yes |

30ms以下は「予約時刻としては間に合う」。しかし、どれもcold投影ではdueにKeyDownできない。Stock窓に入る可能性はあるが、実際のFingering候補、軸状態、hand capacity、既存予約との無競合が確認されていない。このため`SHORTER_LEAD_MOTOR_FEASIBLE`や`HUD_ONLY_SAFE_INTERVENTION`にはしない。

3.180sは数値BPMが初めて画面に出るframeである。ここで正確なdueまで復元できることは未証明なので、同じ表をOracle仮定として別に記録した。30/20/10/0msは時間条件を満たすが、観測条件を満たさない。44msは3.180sでも期限3.176588272sを過ぎている。

Phase 23の反事実も維持した。3.170sまで318 HUD frameの最大差は0、別世界のdueは3.529411805sである。期限前の単一予約は両世界に安全ではない。

## C. NoFail遅延打鍵

Event 36のNoFail自動遷移threshold crossingは`3.249473567s`。10ms仮想Updateのphaseを0–9msまで1ms刻みで固定列挙した。

- 3.251s候補は10/10位相で、Updateの`CheckPostHoldFail`後にEvent 37へ作用し`VeryLate`。
- 3.250s候補も10/10位相でEvent 37へ作用。ただし入力が次Updateまでqueueされるという参照adapter契約に依存する。
- 境界直前3.249sは9/10位相でEvent 37、phase=9msだけEvent 36 `TooLate`。Update/Input順序への感度を実測した反例である。
- 通常Failでは3.251s候補は10/10位相でEvent 36 `TooLate`となり、後に自動Failで停止する。
- NoFailでEvent 37を埋めても、後のID 42/47由来の欠落でずれが再発する。3.251s全10位相でbase mismatchは46、Event 89は自動Miss。Event 37単独救済ではEvent 89を救えない。

この参照モデルのframe内順序は、監査DLLのtime/angle update → `CheckPostHoldFail` → queued KeyDownである。ただしlive UnityのUpdate位相、入力callbackがどのUpdateに入るか、実際のcounter表示順序は未確認である。累積counterだけでは同一capture内のauto FailMissとKeyDown judgementの順序も復元できない。

## D. 実機Stockの最小検証計画

### 1. build同一性

インストール済みDLLをユーザー環境で一度だけhashし、監査対象と比較する。

```powershell
Get-FileHash -Algorithm SHA256 '<installed ADOFAI path>\A Dance of Fire and Ice_Data\Managed\Assembly-CSharp.dll'
```

監査済みSHA256は`0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`。不一致なら現行runtime parityは別監査が必要である。

### 2. Event 36/37だけの撮影

NoFail ON/OFFを分け、同じ譜面・速度・入力装置でEvent 36/37周辺だけを120fps以上で録画する。画面下部の判定counter全体、ゲーム停止、床状態を同時に含める。別途KeyDown timestampを保存し、videoとの同期anchorを2点置く。3.251sを安全値として固定せず、自動遷移境界を挟む入力を複数回記録する。

保存必須項目はgame build、DLL hash、NoFail設定、実測frame time、raw video、入力時計、同期誤差、capture latency区間、counter label、auto FailMissが表示されたframe、停止frameである。

映像だけでUpdate内部順序を確定できない場合は、評価専用Modで`Time.frameCount`、`CheckPostHoldFail`、input dispatch、`AddHit`、counter更新の順序を記録する。その内部ログはruntime parity評価専用でありStudent入力にはしない。

## 再実行

```powershell
uv run pytest -q tests/test_motor_lead_feasibility_v01.py tests/test_audit_phase24_motor_lead_stock_transition_v01.py
uv run python -u scripts/audit_phase24_motor_lead_stock_transition_v01.py --output reports/phase24-motor-lead-stock-transition-<new-run>.json
```

正規成果物は`reports/phase24-motor-lead-stock-transition-20261022-r2.json`。`r1`は12軸全軸のcold latency範囲を追加する前の保存済み予備出力であり、上書き防止規則に従って残した。実機試験、production変更、学習、Validation20、FINAL、Train16全体Motor A/Bは実施していない。
