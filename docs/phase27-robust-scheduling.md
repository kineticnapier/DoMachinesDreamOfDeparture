# Phase 27 — Robust Scheduling & Floor-State Planning

## 判定

`ROBUST_SCHEDULER_VALIDATED`

これは「安全候補を正しく拒否できるprototype契約」の検証であり、Event 37を救済できたという意味ではない。

研究用schedulerはdue仮説、Stock floor-state仮説、Motor到達区間、既存予約、run modeをすべて集合のまま受け取る。全dueの許容窓がMotor区間を包含し、全floor仮説が同じ意図eventを指し、予約競合とbaseline Hitリスクがなく、run modeが一致する場合だけsafe candidateを返す。

仮説なし、Motor envelopeなし、unsupported floor、非因果dueを含む場合はUNKNOWN。反証済み条件や競合がある場合はABSTAINと理由を返す。仮説切捨てによる一意化は行わない。

Event 37の44ms期限前は、Phase 23の2世界の許容窓が非交差で、Event 36 pending/auto-advancedの両floor仮説も残るためABSTAIN/UNKNOWNとなる。3.190sのpoint forecast後も、Event 37軸とexact Motor snapshotが存在せず、floor帰属が未確定なのでUNKNOWNである。

prototypeは予約候補を返すだけでMotor command APIを持たず、productionには接続していない。
