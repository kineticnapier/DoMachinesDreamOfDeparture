# Countdown Perception HUD v3 — Phase 1

## 結論

DLLで確認済みの標準Countdown表示を、描画側とStudent観測側を分離した
HUD v3として実装した。通常開始の制御fixtureでは、旧HUD263で同一だった
Countdown 3回と5回を、最初の数字表示で区別できた。44 msのMotor Lead条件も
このfixtureでは十分に満たす。ただし、これはCountdown表示だけから初回打鍵時刻を
一意に求められるという結論ではない。

機械可読結果は
`reports/countdown-hud-v3-phase1-20261018-r1.json` に保存した。

## 観測契約

HUD v3は既存263次元を変更せず、末尾に15次元を追加する。

- `valid`, `visible`
- `Ready`, `Number`, `Go`
- 数字0〜9のone-hot

観測にはstep、因果的な`observed_at_s`、画面生成時刻`rendered_at_s`、有効性、
表示種別、実際に表示された数字だけを含める。Countdown回数、次の表示時刻、
初回target時刻は含めない。未来のframeはsensorが拒否する。

描画側は内部のCountdown設定とDLL準拠scheduleを使い、`Ready!`、降順の数字、
`Go!`を生成する。Student側はこの描画結果だけを認識する。未対応開始モード、
欠損frame、未知・localize済み・遮蔽された文字列は推測せずinvalid/unknownとする。

## 因果性と反例

| 条件 | 結果 |
|---|---:|
| 旧HUD263のN=3/N=5最大差 | 1.74846e-7（許容5e-5以内） |
| 旧HUD263判定 | `INDISTINGUISHABLE` |
| v3最初の識別時刻 | 0.250 s |
| N=3で観測した数字 | 2 |
| N=5で観測した数字 | 4 |
| N=3の初回打鍵までのLead（評価器のみ） | 2999.500219 ms |
| N=5の初回打鍵までのLead（評価器のみ） | 4999.167029 ms |
| 必要Lead | 44 ms |

観測列のcaptureを完了した後にだけ、評価器が初回target時刻を読み、Leadを照合する。
テストではcapture関数が`targets`、`target_time_s`、`episode_time_s`を参照しないことも
ASTで検査する。

## 初回打鍵時刻を推定できる条件

最初の数字によりCountdown回数の反例は解消する。ただし、絶対時刻の推定には、
少なくとも表示遷移の時刻・周期と、画面に現れる現在速度および床0の幾何が別途必要である。
`Go!`は初回打鍵時刻そのものではない。評価器fixtureでのGoから初回targetまでの間隔は、
120 BPMで500.000010 ms、240 BPMで250.000005 msだった。このため固定秒数を
Studentへ埋め込んではならない。

## 対応範囲と未検証事項

Phase 1が対応するのは標準の通常開始である。次は未検証または明示的にunsupportedである。

- fast takeoff、checkpoint、practice/custom countdown
- localization、遮蔽、OCR誤認識
- 実ゲームの描画・capture遅延とframe drop
- 高BPM、pitch変更、床0幾何を含む網羅的な44 ms Lead評価
- HUD v3からTiming v1.4、Fingering、Motorへの接続

自動fixtureは10 ms観測、capture latency 0 msである。したがって実画面での44 ms保証とは
みなさない。Windows実機では表示frameのtimestampとsensor到着timestampを同時記録し、
この差を差し引いて再評価する必要がある。

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_countdown_hud_v3.py tests/test_audit_countdown_hud_v3_phase1.py tests/test_audit_hud263_nonidentifiability_v01.py tests/test_contact_timing_hud_v01.py --basetemp artifacts/pytest-countdown-hud-v3-phase1-final

.\.venv\Scripts\python.exe scripts/audit_countdown_hud_v3_phase1.py --output reports/countdown-hud-v3-phase1-<new-run-id>.json
```

診断scriptは既存出力を上書きせず終了する。
