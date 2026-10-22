# Phase 30 — DLL Signature Audit

## 判定

`DLL_SIGNATURES_VERIFIED`

この判定はSHA256 `0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`について保存された逆コンパイル資料の署名と制御フローを確認した、という限定的な意味である。インストール済みDLLの一致はユーザー確認済みだが、Codexが実行して確認したものではない。Runtime parityは`RUNTIME_PARITY_UNVERIFIED`のままである。

## 確認した署名

| Method | Visibility / static | Return | Arguments |
|---|---|---|---|
| `scrPlayer.Simulated_PlayerControl_Update` | public instance | `void` | `Nullable<ulong> targetTick = null` |
| `scrPlayer.CheckPostHoldFail` | private instance | `void` | `Nullable<ulong> targetTick = null` |
| `scrPlayer.HitInputEvent` | public instance | `bool` | `bool isAuto = false`, `InputEventState state = Down` |
| `scrPlayer.Die` | public instance | `void` | `bool overload`, `bool multipress`, `string failMessage`, `bool hitbox` |
| `scrController.ProcessKeyInputs` | private instance | `void` | `ulong eventTick` |
| `scrController.PlayerControl_Update` | private instance | `void` | none |
| `scrController.OnPlayerDied` | public instance | `void` | `scrPlayer`と4つのfail引数 |
| `scrController.Fail2Action` | public instance | `void` | none |
| `scrPlanet.SwitchChosen` | public instance | `scrPlanet` | none |
| `scrMarginTracker.AddHit` | public instance | `void` | `HitMargin hit` |
| `scrHitTextManager.ShowHitText` | public instance | `void` | `HitMargin`, `scrPlanet`, `float missAngle = 0` |

Harmony patchはこの完全な引数型を指定する。private methodは名前だけの曖昧探索ではなく、宣言型と引数型を指定した`AccessTools.Method`で取得する。

## 呼出関係

通常の`PlayerControl_Update`はasync inputが無効な場合に`Simulated_PlayerControl_Update()`を呼ぶ。async input経路では`ProcessKeyInputs(ulong)`が対象tickを計算して同じmethodを呼ぶ。

`Simulated_PlayerControl_Update`内の確認済み順序は次の通り。

1. `CheckPostHoldFail`
2. `OttoHoldHit`
3. `HitAutoFloors`
4. `UpdateHoldBehavior`
5. `HitHoldFloorsIfStartedAtHold`
6. `CheckPreHoldFail`
7. `UpdateHoldKeys`

`CheckPostHoldFail`は角度閾値を越えると`Die()`を呼ぶ。NoFailでは`Die`が`noFailInfiniteMargin=true`、`Hit()`、`false`の順で強制遷移し、`SwitchChosen`内で`FailMiss`へ変換する。通常Failでは`OnPlayerDied`から`ChangeState(States.Fail)`へ進み、後に`Fail2Action`が`Fail2`へ遷移する。

`AddHit`は内部累積値を更新する。`ShowHitText`は表示経路であり、両者は同一hookではない。このためPhase 31では別eventとして記録する。

## 未確認

- UMM/Harmonyの実機ロード成功
- hookによるframe schedulingへの影響
- 画面下部カウンターを更新する具体的UI component
- `AddHit`から可視カウンター更新までの実機遅延
- 現行runtimeと参照モデルの一致
