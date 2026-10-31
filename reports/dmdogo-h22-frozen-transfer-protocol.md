# H2.2: 別譜面での固定運指規則の検証

H2.1 の ALL MY FELLAS 初見から得た6区間の左右切替多数派規則を**固定**し、他の譜面へ適用して検証する。

- 入力: H2.1 JSON、別譜面の初見入力ログ（`time_ms Kxx:RI`形式）、同じ別譜面の DMD Motor 実行 JSON（`decisions[].keydowns` を含む）
- 出力: 時刻対応数、未対応数、入力間隔別の左右切替一致率、固定規則と DMD 実行の比較
- 区間: 20ms以下、20–50、50–100、100–150、150–300、300ms超
- **H2.1で学習した ALL MY FELLAS は評価対象として拒否**。規則を再学習しない
- 人間の生ログはGitHubに保存しない。欠測・チャタリングを勝手に削除しない
- 入力時刻の対応は推定値。ADOFAIの判定や人間の好みの証拠ではない
- **現時点で独立した人間の別譜面初見ログは未取得**。未知譜面の一致率は未計測
- 複数キー同時押しや不完全な Motor 実行記録は、この初期版の評価対象外

```powershell
uv run python scripts/audit_dmdogo_h22_frozen_transfer.py `
  --h21-report reports/dmdogo-h21-human-transition-rules-r1.json `
  --human-log "初見入力ログ.txt" `
  --motor-json "DMD実行結果.json" `
  --output reports/dmdogo-h22-transfer-r1.json

uv run pytest -q tests/test_dmdogo_h22_frozen_transfer.py
```

実データを用いた初回の結果が出るまでは、DMDの選択器・Timing・Motorを変更しない。FINAL/Validationのデータにも触れない。