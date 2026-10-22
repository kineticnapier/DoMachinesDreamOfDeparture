# Phase 26 — State-Aware Motor Latency

## 判定

`STATE_AWARE_LATENCY_VALIDATED`（宣言した凍結シミュレータ範囲のみ）

exactな12軸Motor snapshotを複製し、対象軸+1、他軸0を維持するforward rolloutを実装した。指令が現在control tickで反映される場合と、1 tick遅れる場合を举端として、earliest/nominal/latest/uncertaintyを返す。

12軸×cold/preactivated/braked/reversingの48固定caseでは、独立に進めた即時指令の実KeyDownを48/48区間包含した。reversing状態では1 control tickの発行差が接触の離散交差をさらに1 physics tick動かし、区間幅は最大11msとなった。これは同じ凍結物理モデル内の検証であり、実機hardwareの包含保証ではない。

固定44ms、軸別cold、state rolloutを区別した。cold latencyは軸ごとに異なり、動作中状態では位置・速度・activation・fatigue・hand coordinationにより変化する。単一のcold値で状態を代表しない。

1 tick発行遅延まで含むlatest arrivalを44ms以内に収めたのは12/48 caseだけだった。これは44ms production実績を否定する結果ではなく、今回の区間が「現在から最長1 tick指令を遅らせ、その後cold/reversing状態から押す」という別の問いを評価しているためである。44msを状態非依存の物理上界として扱えないことは確認できた。

次の場合はUNKNOWNを返す。

- full snapshotがない
- 予約・他指令の無競合が証明されない
- 軸が既にpressedで、次KeyDownにrelease/reset cycleが必要
- 宣言した指令保持・topology条件から外れる
- bounded rollout内にKeyDownがない

Event 37はproductionがFingering前に棄権したため選択軸とそのsnapshotが存在せず、state-aware Motor feasibilityはUNKNOWNのままである。44ms production契約は変更していない。
