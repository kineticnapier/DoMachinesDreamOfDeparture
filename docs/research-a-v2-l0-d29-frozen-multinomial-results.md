# 研究A v2-L0-D29 — step413の乱数固定抽選・確率交換の測定結果

**判定：`valid_frozen_multinomial_replay`（SUCCESS）。学習改善は確認できない。** D28で残った「PPO更新前後の乱数状態が違うためnone→leftになった」という説明は、この保存済みA0試行では否定された。**step413直前の乱数状態はビット単位で同一であり、確率分布だけを交換すると抽選結果がnone(0)↔left(2)と変化した。**

## 事前登録と独立性
- 解析・検査・測定より前に[固定プロトコル](research-a-v2-l0-d29-frozen-multinomial-protocol.md)を[コミット `6464ad65`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/6464ad65204797f555eda0210d2ec2c59ef2442b)。条件・停止・成立判定は事後変更していない。
- [独立Draft PR #72](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/72)（base D28 PR #71のブランチ）。監査PR #34・基準点PR #44・D28以前の研究PRには書き込んでいない。
- D22原本[Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995) / artifact #13175252296 の `result.json`・`steps.json` を読み、D28原本・陰性対照検査に合格。
- A0単指、seed320403、240ms間隔、開始1.075秒、16目標、480制御step。Python 3.11、torch 2.5.1+cpu、1スレッド、`ATEN_CPU_CAPABILITY`未設定。

## 全step抽選と乱数状態
- `torch.multinomial(torch.tensor(保存済み4確率, dtype=torch.float32), 1, generator=seed320403)` を、B/A→none各480stepでちょうど1回ずつ再生。
- **B：480/480抽選一致、A→none：480/480抽選一致（合計960/960）。** step362の介入は元の抽選を変更しないので、照合対象は `sampled_action`。
- step413の抽選直前に複製した乱数生成器状態は**完全一致**。両系列の状態SHA256：`c1bd5b3f4fec41b591d6273b770215a0be65afd30104a82cbf5d16e395cd7c3e`。
- 事前登録した2×2の確率交換を、各抽選で乱数状態を元に戻して実施：

| step413直前の乱数状態 | 更新前Bの確率を使う | 更新後A→noneの確率を使う |
|---|---|---|
| Bから保存 | **none (0)** | **left (2)** |
| A→noneから保存 | **none (0)** | **left (2)** |

- step413の保存済み確率：B `[0.3233119547, 0.2952822149, 0.2134876102, 0.1679182351]`、A→none `[0.3235397637, 0.2938378751, 0.2145890146, 0.1680332422]`。左指押下（left+both）確率の増加は**0.0012164116**（約0.122パーセントポイント）。最大確率行動は両方none。
- **結論の範囲：** この1試行・このCPU/PyTorch実装・固定乱数状態に限れば、step413の抽選結果の違いは行動確率分布の違いだけで再現できる。乱数列の消費ずれは必要ない。確率差が微小でも、特定の抽選状態では選択行動が変わる。

## 否定的結果と限界
- **「乱数状態のずれがstep413の行動差を作った」という説明は支持されない。** 「argmaxがleftに変化した」という説明も支持されない（両方none）。
- D27ではstep413の単一左指指令を交換すると4.327秒の余分な物理KeyDownが出現・消失した。ただしD29は**抽選のみ**であり、物理再生や新規採点は行っていない。
- D19非介入sampled576目標：PPO更新前→後のHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。**低速打鍵の学習改善は依然として確認できない**。
- PPOがなぜこの方向に確率を変えたか（優位度・行動別勾配・学習信号）は未解決。単一seed・A0のみ。A1二指、A3未学習対照、他seed、人間同等性、単指物理的不可能性、正式研究002承認は未証明。左右交代率・使用率は成功条件に使用していない。
- 新規学習0、PPO/Adam更新0、方策推論0、Motor再生0、物理KeyDown0、Hit/Extra新規採点0。

## 検査・成果物
- 新規解析 `scripts/a002v2_l0_d29_frozen_multinomial_rng.py`、専用検査 `tests/test_research_a_v2_l0_d29_frozen_multinomial_rng.py`、専用Actions。
- [初回成功Actions #41569034433](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41569034433)：**SUCCESS、9 passed、1 warning**。警告はNumPy未導入によるPyTorch初期化警告で、抽選・結果照合は成功。Node.jsの非推奨警告も記録。
- [結果JSON・tests.log・run.log artifact #13201225187](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41569034433/artifacts/13201225187)（30日保持）。結果の主要数値はこの文書にも永続保存。
- **技術的再生の成立は学習・運指獲得の成功を意味しない。** 自動マージ・Draft解除・既存PRマージは禁止。
