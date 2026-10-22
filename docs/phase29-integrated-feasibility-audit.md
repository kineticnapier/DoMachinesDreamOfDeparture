# Phase 29 — Integrated Feasibility Audit

## 最終判定

- `RUNTIME_PARITY_UNVERIFIED`
- `STATE_AWARE_LATENCY_VALIDATED`（凍結シミュレータ・exact snapshot条件）
- `ROBUST_SCHEDULER_VALIDATED`（安全な拒否契約）
- `ORACLE_ONLY_INTERVENTION`
- `NO_SAFE_INTERVENTION_ESTABLISHED`
- `INSUFFICIENT_EVIDENCE`

## 統合結果

保存済みPhase 16物理結果はbaseline 162/220 Hit、candidate 207/220 Hitで、新規Hit 46、消失Hit 1（Event 89）、純増45だった。この既存非回帰FAILは変更していない。

Phase 26〜28はproduction Motor A/Bを実行していない。したがってHit/Missをnullのまま保持した。State-aware envelopeは48/48 containment、Phase 27はEvent 37についてsafe 0、UNKNOWN 2、false-safe 0。Phase 28は保存済み5介入のうちEvent 89を回復したものがOracle全補完1件、HUD-only safe 0、false-safe 0だった。

保存済みNoFail Stock参照ではbaseline floor attribution mismatch 50、通常Fail参照は停止、NoFail参照は継続する。live runtime parityは未検証である。

## Event 89

Event 89のPerfect回復に必要だった上限条件はEvent 36・37・42・47の真値時刻入力である。これは予約期限までにHUD-onlyで選択できず、実軸Motor envelope、floor-state一意性、baseline Hit保存、live Stock parityも欠ける。Event 89はOracleでは回復したが、HUD-onlyでは回復していない。

## 採用を止める制約

1. 44ms期限前のEvent 37反事実二世界はHUD prefixが一致し、Stock窓が非交差。
2. Event 36 pending/auto-advancedのfloor仮説が介入時に一意でない。
3. Event 37はproductionがFingering前に棄権したため、選択軸とexact Motor snapshotがない。
4. Event 37だけを埋めても42・47でずれが再発する。
5. 実機DLL identity、Update/Input/render順序、auto FailMiss counter表示が未確認。
6. 保存済みPhase 16 candidateはbaseline Event 89 Hitを失う。

次世代production制御へ進む条件は、Phase 25の実機trace、実際のEvent 37候補軸snapshotを使ったPhase 26 envelope、可視feedbackでfloor仮説を期限前に絞れる証拠、そして既存Hitを失わない限定Motor A/Bである。

## 回帰結果

- Phase 16〜29関連: `113 passed in 9.60s`
- repository全体: `2079 passed, 1 warning in 180.36s`
- 実行: `uv run pytest -q -o "pythonpath=src ."`
- warning: 既存`tests/test_onset_ablation.py`のPyTorch tensor→scalar変換

production Timing/Motor/Fingering/Scorer/HUD、checkpoint、training設定は変更していない。Validation20、FINAL、Train16全体Motor A/B、学習は実行していない。

## Phase commits

- Phase 25: `cc4ce67`
- Phase 26: `83d4102`
- Phase 27: `4fef3d0`
- Phase 28: `40bfc21`
- Phase 29: `d66e617`

## 次の実機作業

最初に必要なのは学習やMotor A/Bではなく、インストール済みDLL identityの取得である。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/collect_phase25_stock_runtime_identity.ps1 `
  -AssemblyCSharpPath '<ADOFAI>\A Dance of Fire and Ice_Data\Managed\Assembly-CSharp.dll' `
  -OutputPath 'reports/phase25-installed-dll-identity-<run>.json'
```

一致後もruntime parityは未確定である。Phase 25文書のhook契約に従うEvent 36/37限定traceが次の証拠となる。
