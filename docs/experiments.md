# 実験記録 / Experiment Log

この文書には、身体モデル・学習条件・結果の変更履歴を記録する。

This document records experiments, body-model changes, training conditions, and results.

実験結果だけでなく、失敗した設定と過程も残す。

Record failed configurations and the process leading to a result, not only successful outcomes.

---

## Experiment 000 — Baseline

**状態 / Status:** Not started

### 目的 / Objective

2本指身体モデルと簡易リズム環境が決定論的に動作し、同じ初期状態・同じ行動列から同じ入力結果を再現できることを確認する。

Verify that the two-finger body model and simplified rhythm environment behave deterministically: the same initial state and action sequence must reproduce the same input events.

### 条件 / Conditions

TBD

### 結果 / Results

TBD

### 観察 / Observations

TBD

---

## Experiment 001 — Single Finger Limit

**状態 / Status:** Not started

### 目的 / Objective

身体モデルにおける単指連打の速度限界を測定し、v0.1 の交互押し実験に使う速度域を決定する。

Measure the single-finger tapping limit of the body model and determine the rate range for the v0.1 alternation experiment.

### 条件 / Conditions

TBD

### 結果 / Results

TBD

### 観察 / Observations

TBD

---

## Experiment 002 — Emergent Alternation

**状態 / Status:** Not started

### 目的 / Objective

単指では物理的に突破できない入力列に対して、運指を教えずに学習したエージェントが2本の指を利用する戦略を自力で獲得するか検証する。

Test whether an agent trained without fingering instructions independently learns to use both fingers on a sequence that is physically impossible to clear with one finger.

### 成功条件 / Success Criterion

- 単指方策では突破不能 / a single-finger strategy cannot clear the sequence
- 運指を指定する報酬が存在しない / no reward term specifies a fingering pattern
- 学習後の方策が安定して突破する / the learned policy clears the sequence consistently
- 行動履歴から獲得した運指を再現できる / the acquired fingering can be reconstructed from logs

### 条件 / Conditions

TBD

### 結果 / Results

TBD

### 観察 / Observations

TBD
