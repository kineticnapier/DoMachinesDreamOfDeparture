# D18 実行記録（結果未確定）

- 実行前の事前登録: `8dc4d90cbb0b3d4bebdd0c16016952533ae03be3`。変更禁止。
- 独立Draft PR #61、base D17 PR #60。D8/D12保存済み成果物の再利用のみ。追加学習・Motor再実行は行わない。
- この記録は結果ではない。専用Actionsの実行結果を確認するまで成功・失敗を断定しない。
- 判定基準は事前登録に従う。失敗時は失敗ログを保持し、既存モデル・事前登録を変更しない。

## 初回実行の否定的結果（2026-11-03 JST）
- [Actions #41476343280](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476343280) と [#41476416807](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476416807) は **FAILURE**。最初のrunでは9テスト成功、1警告。D8/D12保存物取得成功後、確率比較が `KL unexpectedly negative` で停止。実測比較は未確定。初回失敗の [artifact #13157377925](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476343280/artifacts/13157377925) に `result.json`、`tests.log`、`run.log` を保存。
- 原因：float32で計算した確率の行和が厳密な1とは限らず、そのままfloat64でKLを計算すると負の丸め誤差を生じうる。**事前登録の条件・判定基準は変更しない。** KLの計算だけ行ごとにfloat64で正規化してから実施し、元のfloat32確率・平均確率差・argmaxは変更しない。専用の回帰検査を追加。
- この段階では新規学習・PPO更新・Motor実行0、Hit改善なし。修正後の結果は未確認。
