# Phase 32 — Runtime Trace Ingestion

`stock_runtime_trace_v02`はPhase 31 JSONLをsession単位で読み、sequence gap/duplicate/reversal、frame逆行、timestamp逆行、欠損hook、NoFail状態を監査する。複数headerを連結したcaptureも混同せず分離する。

抽出結果は自動FailMiss候補、KeyDownから次の判定hookへの対応、`AddHit`から`ShowHitText`までのsignal遅延を含む。floor IDと内部counterはEvaluator-onlyのままで、Student向け変換APIはない。

合成fixtureではNoFailと通常Failの2 sessionを検証した。これはparserの証拠であって実機証拠ではないため、判定は`RUNTIME_PARITY_UNVERIFIED`である。また`ShowHitText`は可視判定文字のsignalであり、画面下部の累積counter更新そのものではない。累積counter遅延は`UNKNOWN_NOT_HOOKED`として保持する。

実機ログ取込:

```powershell
uv run python scripts/audit_phase32_runtime_trace_ingestion_v01.py `
  --input '<Mod>\runtime-traces\stock-runtime-<session>.jsonl' `
  --output 'reports\phase32-runtime-trace-live-<run>.json'
```
