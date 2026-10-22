# Phase 34 — Integrated Runtime Evidence Audit

## 判定

- `DLL_SIGNATURES_VERIFIED`
- `RUNTIME_TRACE_MOD_BUILT`
- `RUNTIME_TRACE_MOD_UNVERIFIED`
- `RUNTIME_PARITY_UNVERIFIED`
- `STOCK_VISIBLE_FEEDBACK_PARTIAL`
- `INSUFFICIENT_EVIDENCE`

Phase 30は監査済み逆コンパイル資料上の署名と呼出順を固定した。Phase 31 Modはインストール済み参照を用いて.NET Framework 4.8向けに0 warningでビルドした。Phase 32は複数session・欠損・重複・順序・NoFail分離を処理する。Phase 33はUnity timestampを保持してPhase 20Cと比較し、10ms仮想gridを実機時刻へ強制しない。

一方、実機でModをloadした証拠、Harmony hook発火ログ、Event 36/37/42/47/89のruntime traceはまだない。`ShowHitText`は可視signalだが、画面下部の累積counter componentと更新時刻は未特定である。このためStock参照モデルの一致・不一致はいずれも確定していない。

## 非干渉性

Mod hookはread-only Prefix/Postfixであり、引数・戻り値・ゲーム状態を変更しない。ログ例外を隔離し、bounded queueとbackground writerを使う。内部floor、song time、内部counterはEvaluator-only JSONLにだけ存在し、Studentやproduction HUDへの経路はない。

production Timing、Motor、Fingering、Scorer、HUD、checkpoint、training設定は変更していない。Validation20、FINAL、Train16全体Motor A/B、学習も実行していない。

## 最小実機手順

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_phase31_stock_runtime_probe.ps1 `
  -OutputDirectory artifacts/phase31-stock-runtime-probe-build-<run>
```

生成物を`<ADOFAI>\Mods\DmdodStockRuntimeProbe`へコピーし、`probe-config.json`で34〜40を指定してNoFail/通常Failを別sessionとして記録する。次にPhase 32、33のscriptへJSONLを渡す。Event 42/47と89は範囲を分けた追加sessionにし、ログ負荷を限定する。

実機capture後も、sequence gap、必須hook欠損、floor map欠落、累積counter未観測のいずれかがあれば完全Parityは保留する。
