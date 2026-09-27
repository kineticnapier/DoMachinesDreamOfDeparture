# Experiment Log

This document records experiments, body-model changes, training conditions, and results.

Record failed configurations and the process leading to a result, not only successful outcomes.

---

## Experiment 000 — Baseline

**Status:** Not started

### Objective

Verify that the two-finger body model and simplified rhythm environment behave deterministically: the same initial state and action sequence must reproduce the same input events.

### Conditions

TBD

### Results

TBD

### Observations

TBD

---

## Experiment 001 — Single Finger Limit

**Status:** Not started

### Objective

Measure the single-finger tapping limit of the body model and determine the rate range for the v0.1 alternation experiment.

### Conditions

TBD

### Results

TBD

### Observations

TBD

---

## Experiment 002 — Emergent Alternation

**Status:** Not started

### Objective

Test whether an agent trained without fingering instructions independently learns to use both fingers on a sequence that is physically impossible to clear with one finger.

### Success Criterion

- A single-finger strategy cannot clear the sequence.
- No reward term specifies a fingering pattern.
- The learned policy clears the sequence consistently.
- The acquired fingering can be reconstructed from logs.

### Conditions

TBD

### Results

TBD

### Observations

TBD

[日本語](実験記録.md)
