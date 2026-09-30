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

---

## Experiment 003 — v0.9 BC Training Acceleration

**Status:** Complete

### Objective

Reduce per-round Behavioral Cloning (BC) cost in the v0.9 press-persistence trainer while preserving the training semantics as closely as possible.

### Conditions

- Dataset: `data/DMDOD-v090-tuf`
- Hidden size: 128
- Chunk size: 192 steps
- Schedule: 48 rounds × 12 epochs
- After convergence, fixed-point skipping leaves only two heavy epochs in many rounds
- CPU: Ryzen 5 5600G
- GPU: RTX 3060 12GB
- CPU and CUDA floating-point kernels are not bit-identical, so the CUDA backend is validated by numerical closeness rather than byte identity

### Process

1. Replaced the frame-by-frame `GRUCell` Python loop inside each BC chunk with `forward_sequence()` and a fused GRU invocation.
2. Preserved sequence order, chunk boundaries, hidden-state detach points, loss, gradient clipping, and Adam step count.
3. Measured reverse BC at roughly 88–93 seconds across multiple rounds on CPU.
4. The earlier heavy epoch was about 223.5 seconds; the overall heavy path therefore became substantially shorter, although that number is not an identical BC-only measurement.
5. Profiled one 93.15-second reverse BC epoch:
   - forward: 25.18 s
   - loss: 2.85 s
   - backward: 60.94 s
   - optimizer: 2.64 s
   - other: 1.55 s
6. After Python overhead became minor, GRU/autograd backward was identified as the dominant remaining BC bottleneck.
7. Added a CUDA backend for reverse BC only. The trusted CPU model is not mutated directly; training occurs on a CUDA copy and is committed back only after the epoch succeeds, allowing CPU fallback after CUDA failure.
8. The initial environment had `torch 2.14.0+cpu` and `torch.cuda.is_available() == False`, so the CUDA hardware test was skipped. After installing a CUDA-enabled wheel, the CUDA comparison test reported 3 passed.
9. The CUDA test produced a cuDNN warning that GRU weights were not stored in one contiguous chunk. The backend was changed to use a temporary packed `nn.GRU` to avoid repeated repacking while leaving checkpoint/state-dict structure unchanged.
10. Benchmarked the packed-GRU CUDA backend in production round 37. Reverse BC completed 1516 chunks in 11.29 seconds (setup 0.18 s, train 11.11 s, copyback 0.00 s). This first measurement had 0 CUDA sequence-cache hits and 96 misses.
11. In the same run, the CPU forward prefix miss took 88.52 seconds, while the next forward prefix hit after round 37 took 2.34 seconds. The 2m26s total time for round 37 was therefore dominated by rebuilding the forward prefix cache after process restart.
12. From round 38 onward, the CUDA sequence cache behaved as expected at 94 hits / 2 misses.
13. Across rounds 38–48, cache-hit reverse BC ranged from 9.73 to 11.89 seconds and averaged about 10.41 seconds, approximately **8.95× faster** than the 93.15-second CPU baseline.
14. With light Guard evaluation, total round time fell into the 16–19 second range. On rounds such as 39, 41, 43, 47, and 48, Guard evaluation became the dominant cost.
15. The final resumed session started from completed round 36 and finished round 48 with a displayed elapsed time of 12m47s.
16. Final cache statistics were eval-cache hits=1238 / misses=731, BC proposal cache hits=0 / misses=24, teacher cache hit=94 / generated=0.

### Results

- Batched-GRU CPU path: effective; reverse BC is stable at roughly 88–93 seconds.
- CPU profile: approximately 65% of reverse BC time is backward and 27% is forward.
- CUDA backend: CPU/CUDA numerical-closeness test passed.
- Packed-GRU CUDA production benchmark: reverse BC 93.15 s → 11.29 s, approximately **8.25× faster**.
- After CUDA sequence-cache warm-up (rounds 38–48): 9.73–11.89 s, average about 10.41 s; approximately **8.95× faster** than the 93.15-second CPU baseline.
- With a light Guard, total round time fell to roughly 16–19 seconds.
- After CUDA acceleration, Guard evaluation became the main bottleneck rather than BC.

### Observations

- After removing the Python frame loop, `other` time fell to about 1.55 seconds, so further C++-only loop migration is unlikely to provide a large gain.
- Forward BC falls to roughly 1–2.3 seconds when the immutable expert-prefix cache hits.
- Reverse BC starts with round-local sequences, so the same fixed-prefix cache cannot be applied directly.
- The CUDA sequence cache reuses the 94 fixed trajectories within a process and reaches `94hit/2miss` in steady state.
- In round 39, forward BC was 1.93 s and reverse CUDA BC 11.90 s, while Guard totaled 81.74 s.
- In round 48, Guard still consumed 42.81 s + 10.42 s, so expensive Guard rounds remained present at the end of training.

---

## Experiment 004 — v0.9 Final Generalization / Holdout

**Status:** Complete

### Objective

Evaluate the v0.9 press-persistence policy trained on 48 train charts using 10 validation charts and 2 final holdout charts that were completely excluded from round selection.

### Conditions

- Train: 48 charts
- Validation: 10 charts
- Final holdout: `Celica`, `Plum - Aftershine`
- FINAL charts are never evaluated during round selection or trust-region Guard
- Checkpoint: `checkpoints/real_chart_v090_press_persistence.pt`

### Results

- Train anchors aggregate: H7392/7686, meanX95.8%, overload=False
- Validation aggregate: H4438/4466, meanX97.9%, overload=False
- FINAL holdout aggregate: H1168/1202, meanX96.8%, overload=False

`Celica`:

- H=431/434, miss=3
- X=98.89%, PP=98.6%
- MAE=24.67 ms
- early=2, overload=False, keydowns=433

`Plum - Aftershine`:

- H=737/768, miss=31
- X=94.64%, PP=94.2%
- MAE=27.87 ms
- early=12, overload=False, keydowns=749

### Observations

- The 10 validation charts completed at H4438/4466 aggregate, meanX97.9%, with no overload.
- Neither final holdout chart overloaded.
- `Celica` reached 431/434 hits with very high timing accuracy.
- `Plum - Aftershine` remained the harder holdout: 737/768 hits with 31 misses.
- The final checkpoint was written after completion of round 48.

[日本語](実験記録.md)
