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

**Status:** In progress

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

### Results

- Batched-GRU CPU path: effective; reverse BC is stable at roughly 88–93 seconds.
- CPU profile: approximately 65% of reverse BC time is backward and 27% is forward.
- CUDA backend: CPU/CUDA numerical-closeness test passed.
- Production speed after the packed-GRU warning fix: **not measured yet**.

### Observations

- After removing the Python frame loop, `other` time fell to about 1.55 seconds, so further C++-only loop migration is unlikely to provide a large gain.
- Forward BC falls to roughly 1.8–2.0 seconds when the immutable expert-prefix cache hits.
- Reverse BC starts with round-local sequences, so the same fixed-prefix cache cannot be applied directly.
- On difficult rounds, Guard can rise to roughly 35–50 seconds and becomes the second major bottleneck outside BC.
- Re-measure production CUDA speed after the packed-GRU change and append the result here.

[日本語](実験記録.md)
