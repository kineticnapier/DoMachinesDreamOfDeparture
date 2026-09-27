# v0.1 仕様 / Specification

## 1. 目的 / Objective

人間と同等の知覚・身体的制約を与えられた学習機械が、ADOFAI において人間とは異なる有効な運指を獲得できるかを調べる。

Investigate whether a learning agent under human-like perceptual and physical constraints can acquire effective ADOFAI fingering strategies different from those used by humans.

「最強の Bot」を作ることは目的ではない。身体能力を可能な限り固定し、学習・予測・運指の違いを調べる。

The objective is not to build the strongest possible bot. Physical ability should be controlled as far as practical so that differences in learning, prediction, and fingering can be studied.

## 2. 構成 / Architecture

```text
ADOFAI Environment
        |
   perception
        v
      Agent
        |
  motor command
        v
 Neural Delay
        |
        v
   Hand Model
        |
        v
 Keyboard Model
        |
        v
      Input
```

以下を独立した部品として実装する。

The following components should remain independently replaceable:

- Agent
- Perception
- Hand Model
- Keyboard Model
- Environment
- Logger / Evaluator

## 3. 禁止する情報・操作 / Forbidden Information and Actions

本番条件では、エージェントは以下を利用できない。

Under evaluation conditions, the agent must not:

- `.adofai` から正解入力時刻を直接取得する / read exact target input times from `.adofai`
- ゲーム内部の beat や判定予定時刻を取得する / read internal beat or judgement timing
- Otto / Autoplay の情報を利用する / use Otto or Autoplay information
- ゲーム速度・判定幅を変更する / alter game speed or judgement windows
- キー入力を直接発火する / directly emit a key press

エージェントが出力できるのは身体への運動指令のみとする。

The agent may output only motor commands to the simulated body.

## 4. 時間 / Time

身体シミュレーションは固定刻みとし、初期値を

The body simulation uses a fixed time step, initially

```text
Δt = 1 ms (1000 Hz)
```

とする。方策の更新周期はこれと分離し、初期値を 100 Hz とする。

The policy update frequency is independent of the body simulation and initially set to 100 Hz.

## 5. 手モデル v0.1 / Hand Model v0.1

最初は2本指のみを扱う。

The first model has two fingers only:

```text
Left
Right
```

各指 `i` の状態を

Each finger `i` has state

```text
S_i = (x_i, v_i, a_i, f_i)
```

とする。

- `x_i`: 指の位置 / finger position
- `v_i`: 指の速度 / finger velocity
- `a_i`: 筋活動 / muscle activation
- `f_i`: 疲労 / fatigue

エージェントの行動は

The agent action is

```text
u_i ∈ [-1, 1]
```

であり、`+1` は最大限押す方向、`0` は能動的な力なし、`-1` は最大限戻す方向を表す。

where `+1` means maximum downward command, `0` no active command, and `-1` maximum upward command.

## 6. 筋肉応答 / Muscle Response

運動指令を即座に力へ変換しない。

Motor commands do not become force instantaneously.

```text
τ_a da_i/dt = u_i - a_i
```

利用可能な力を

Available force is modeled as

```text
F_i = F_max (1 - f_i) a_i
```

とし、簡略化した指の運動を

and simplified finger motion as

```text
m_i d²x_i/dt² = F_i - c_i dx_i/dt - k_i(x_i - x_0,i)
```

とする。各式は人体の確定モデルではなく、実測結果に合わせて交換・調整する対象である。

These equations are not claimed to be a definitive physiological model. They are replaceable approximations whose parameters and structure should be validated against measurements.

## 7. キーボード / Keyboard

キーには作動点と復帰点を持たせる。

A key has separate actuation and reset points. Initial example values are:

```text
travel      = 4.0 mm
actuation   = 2.0 mm
reset       = 1.8 mm
```

下降中に作動点を通過したときのみ `KEY_DOWN` が成立し、復帰点まで戻らなければ次の入力は成立しない。

`KEY_DOWN` occurs only when the key crosses the actuation point downward, and another press cannot occur until the key returns past the reset point.

## 8. 疲労 / Fatigue

v0.1 の暫定モデル：

Temporary v0.1 model:

```text
df_i/dt = α |a_i| - β f_i
0 <= f_i <= 1
```

疲労によって利用可能な最大筋力を低下させる。ただし疲労式は将来の実測によって変更する。

Fatigue reduces available force. The fatigue model is explicitly provisional and should be replaced or calibrated using measurements.

## 9. 指同士の干渉 / Finger Coupling

2本の指を完全独立とはしない。v0.1 では簡略化した干渉を導入する。

The two fingers are not treated as perfectly independent. v0.1 uses a simple coupling term, for example:

```text
F'_i = F_i (1 - γ |a_j|)
```

将来10指へ拡張する場合は、指ごとの干渉を行列などで表現できる設計にする。

A future ten-finger model should support finger-specific coupling, for example through a coupling matrix.

## 10. 知覚 / Perception

### Mode A: Symbolic

開発・身体モデル検証用。現在角度、次の床の相対情報、回転速度などの抽象化した状態を与える。正解入力時刻そのものは与えない。

Used for development and body-model validation. The agent receives abstract state such as current angle, relative information about upcoming tiles, and rotation speed, but never exact target input times.

### Mode B: Visual

本番用。ゲーム画面から知覚する。フレーム更新周期と知覚遅延を持たせる。

Used for final evaluation. Perception comes from rendered game frames and includes a finite frame rate and perceptual delay.

```text
observation at t = I(t - τ_visual)
```

未来を内部で予測することは禁止しない。人間も先読みと運動準備を行うためである。

Internal prediction of future events is allowed; humans also anticipate upcoming events and prepare movements in advance.

## 11. 強化学習 / Reinforcement Learning

初期実装では連続行動を扱える手法を使用する。候補として PPO を想定するが、方式は固定しない。

The initial implementation should use an algorithm supporting continuous actions. PPO is a candidate, but the learning algorithm is not part of the fixed specification.

行動：

Action:

```text
a_t = (u_left, u_right)
```

報酬の基本形：

Initial reward structure:

```text
R = R_hit - λ_t |timing_error| - λ_e E - R_miss
```

消費量の簡略値：

Simplified effort term:

```text
E = Σ a_i²
```

報酬によって特定の運指を直接教えてはならない。

The reward must not directly prescribe a particular fingering pattern.

## 12. 学習段階 / Curriculum

```text
Stage 0  single timed press
Stage 1  constant BPM
Stage 2  increasing BPM
Stage 3  speed where alternating fingers become advantageous
Stage 4  changing BPM
Stage 5  simple ADOFAI patterns
Stage 6  high difficulty
Stage 7  U-level frontier
```

Stage 3 が v0.1 の主要実験となる。交互押しを明示的に教えず、エージェントが自力で獲得するかを見る。

Stage 3 is the main v0.1 experiment. Alternating fingers must not be explicitly taught; the experiment tests whether the agent discovers such a strategy independently.

## 13. 人間モデルの校正 / Human Calibration

人体パラメータを「それらしい値」で決めるだけではなく、人間の測定結果に合わせる。

Body parameters should not be chosen only because they appear plausible. They should be calibrated against human measurements.

候補となる測定：

Candidate measurements include:

- 単指連打 / single-finger tapping
- 2指交互 / two-finger alternation
- 同時押し / simultaneous presses
- 徐々に速度を上げる試験 / gradually increasing rate
- 長時間連打 / sustained tapping
- 不規則タイミング / irregular timing

人間の測定結果を `H_j`、モデルの結果を `M_j(θ)` として、概念的には

With human measurements `H_j` and model measurements `M_j(θ)`, calibration can be expressed conceptually as

```text
θ* = argmin_θ Σ_j w_j D(M_j(θ), H_j)
```

とする。

## 14. 未知試験 / Held-out Validation

校正に使った課題だけで人間に似ていても十分ではない。一部の運指・速度変化を校正に使わず残し、未知条件でも人間と似た性能曲線を示すか検証する。

Matching humans only on calibration tasks is insufficient. Some patterns and timing conditions must be held out and used to test whether the model reproduces human-like performance on unseen tasks.

例：

Example:

```text
Calibration: single finger, LR alternating, sustained tapping
Held out:    LLR, LRR, LLRR, irregular BPM, short bursts
```

## 15. 記録 / Logging

最低限、以下を各試行で保存する。

At minimum, record the following for each run:

```text
timestamp
observation
policy output
finger position
finger velocity
muscle activation
fatigue
key state
target timing
actual timing
timing error
judgement
reward
```

研究用の正解時刻は評価・解析器が保持してよいが、エージェントの観測へ流してはならない。

The evaluator may retain exact target timings for scoring and analysis, but they must never leak into the agent observation.

## 16. v0.1 で実装しないもの / Out of Scope for v0.1

- 10指 / ten fingers
- 手首・腕 / wrist and arm
- 詳細な腱・骨格モデル / detailed tendon and skeletal simulation
- 3D人体 / 3D human body
- 個々の実在筋肉の再現 / individual anatomical muscles
- 本番用視覚入力 / final visual perception
- 音声知覚 / audio perception
- 実ADOFAIへの直接接続 / direct control of the real game

まずは

The initial target is simply

```text
2-finger physical model
+ simplified rhythm environment
+ reinforcement learning agent
```

とする。

## 17. v0.1 成功条件 / v0.1 Success Criterion

単指では身体モデル上物理的に突破できない速度の入力列を用意する。運指に関する説明・報酬を与えず、エージェントが2本の指を利用した周期運動またはそれ以上に有効な戦略を獲得して突破すること。

Create a sequence whose rate is physically impossible for the simulated body to clear with one finger. Without giving fingering instructions or fingering-specific rewards, the agent must discover a two-finger periodic motion or another effective strategy and clear it.

```text
single finger: impossible
learned strategy: clear
```

これを v0.1 の最初の到達点とする。

This is the first milestone of v0.1.
