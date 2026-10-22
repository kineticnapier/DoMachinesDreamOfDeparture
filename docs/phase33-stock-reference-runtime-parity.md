# Phase 33 — Stock Reference Parity

## 判定

`RUNTIME_PARITY_UNVERIFIED`

実機traceがまだ存在しないため、Phase 20C参照モデルとの比較器のみ実装した。Phase 32の合成fixtureを比較しても判定は昇格しない。

比較器はNoFail/通常Failを分離し、runtimeで観測した`CheckPostHoldFail`、KeyDown、`SwitchChosen`、`AddHit`、game stopをPhase 20C transitionと比較する。カテゴリ、floor before/after、target ordinal、時刻を別々に`MATCH`、`MISMATCH`、`UNKNOWN`へ分類する。

実機時刻にはprobeが記録したUnity frame/realtimeと評価専用song timeを用いる。Phase 20Cの10ms仮想gridを実機timestampへ代入しない。floor seqIDからplayable target ordinalへの対応表がない場合、判定対象は`UNKNOWN`であり、他項目が一致しても完全Parityを宣言しない。

`RUNTIME_PARITY_CONFIRMED`には、実機evidence、監査DLL hash、欠損のないsequence、必須hook coverage、target対応、全transition一致がすべて必要である。明示的不一致がある実機qualified traceだけが`RUNTIME_PARITY_MISMATCH`になり、不完全traceは常に`UNVERIFIED`となる。

```powershell
uv run python scripts/audit_phase33_stock_runtime_parity_v01.py `
  --runtime '<Mod>\runtime-traces\stock-runtime-<session>.jsonl' `
  --floor-map 'reports\phase33-floor-map-<session>.json' `
  --output 'reports\phase33-runtime-parity-<run>.json'
```
