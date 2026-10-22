# Phase 25 — Stock Runtime Parity & Instrumentation

## 判定

`RUNTIME_PARITY_UNVERIFIED`

Codex環境には実機ADOFAI runtime logがない。監査DLLと現在のインストールbuildの同一性、UnityのUpdate/Input/render順序は未確認である。合成fixtureはパーサーと順序保持だけを検証し、実機証拠として扱わない。

## 計測契約

JSONLはheaderの後に、単調な`sequence`、`frame`、`realtime_s`を持つeventを記録する。hook点は`frame_begin`、`CheckPostHoldFail`、input dispatch、物理KeyDown、judgement、auto FailMiss、`AddHit`、visible counter update、game stop、`frame_end`である。NoFail状態は各行に固定する。

floor、内部判定対象、frame内呼出順はEvaluator-only payloadである。これはruntime parity検証専用で、Studentへ公開する型や変換関数は実装していない。Studentが利用可能なのは別系統の可視counter captureだけである。

未知のHarmony patch signatureを推測したModは追加していない。実機buildをhashし、逆コンパイルまたはreflectionで正確なsignatureを再確認してから、観測のみのprefix/postfix hookを実装する。

## DLL identity

```powershell
powershell -ExecutionPolicy Bypass -File scripts/collect_phase25_stock_runtime_identity.ps1 `
  -AssemblyCSharpPath '<ADOFAI>\A Dance of Fire and Ice_Data\Managed\Assembly-CSharp.dll' `
  -OutputPath 'reports/phase25-installed-dll-identity-<run>.json'
```

hash一致は必要条件にすぎず、runtime動作一致ではない。

## 実機probeに必要なhook

監査済み資料では`scrPlayer.Simulated_PlayerControl_Update`が`CheckPostHoldFail`を先に呼び、その後input queueを処理する。`scrPlanet.SwitchChosen`、`scrMarginTracker.AddHit`、`scrHitTextManager.ShowHitText`、game stop経路を同じframe/sequence clockで記録する。hookは引数・戻り値・状態を変更せず、例外時もゲーム制御へ介入しないappend-only loggerとする。

## 合成fixture結果

Event 36 auto FailMiss → AddHit → counter fixture → input dispatch → Event 37 VeryLateの順序を保存できた。DLL hash欄は監査値と一致するが、`evidence_kind=synthetic_fixture`なので判定は必ずUNVERIFIEDとなる。
