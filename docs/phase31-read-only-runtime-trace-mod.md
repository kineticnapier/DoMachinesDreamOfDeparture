# Phase 31 — Read-only Runtime Trace Mod

## 判定

- `RUNTIME_TRACE_MOD_BUILT`
- `RUNTIME_TRACE_MOD_UNVERIFIED`

`tools/stock-runtime-probe`はPhase 30で確認した完全なoverloadだけをHarmonyでpatchする。Prefix/Postfixは引数、戻り値、`__runOriginal`、ゲーム状態を変更しない。すべてのhook本体は例外を隔離し、JSON serializationとfile writeはbounded queue経由のbackground threadへ移す。

インストール済みUMM/Harmony参照は.NET Framework 4.8であり、Modも`net48`をtargetとする。`Assembly-CSharp`の基底型依存に必要な`RDTools.dll`も明示参照する。最初の`net472`試行と、RDTools参照前の試行は依存関係検出用の失敗であり採用していない。

## 計測範囲

既定の`probe-config.json`はfloor 34〜40だけを記録する。Event 42・47・89を撮影するときは別sessionで範囲を明示的に変更する。各行はsession ID、frame、Unity realtime、単調sequence、NoFail、hook、floor、judgement、入力、内部counter snapshot、hook所要時間を持つ。

floor IDと`AddHit`内部counterはEvaluator-onlyであり、headerの`student_visible_fields`は空である。`ShowHitText`は可視判定signalとして別eventにする。画面下部counter componentはDLL資料だけでは確定できなかったため、内部counterを可視counterと称していない。

## ビルド

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_phase31_stock_runtime_probe.ps1 `
  -OutputDirectory artifacts/phase31-stock-runtime-probe-build-<run>
```

生成ディレクトリをADOFAIの`Mods\DmdodStockRuntimeProbe`へコピーし、`probe-config.json`の対象floorを確認してからUMMで有効化する。ログはMod配下`runtime-traces`へ出る。Modを無効化するとloggingを止め、unload時にpatchを解除する。

## 限界

Codex側のbuild成功は実機でModがloadされたこと、hookが発火したこと、frame perturbationが許容範囲だったことを意味しない。実機JSONLをPhase 32へ取り込むまでruntime parityは未確認である。
