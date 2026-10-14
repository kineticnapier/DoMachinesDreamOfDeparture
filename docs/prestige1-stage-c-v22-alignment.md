# Prestige I — Stage C v2.2 Train-only Target-Alignment Audit

**Purpose:** Test a specific, evidence-based hypothesis from frozen
Stage C v2.1 Validation failure diagnostics: the Timing head is supervised
on the next chronologically future chart event, while the expert Motor
scheduler and the physical game target the **next unresolved note**. An
early hit can leave the just-hit event in the future schedule even though
the physical next note has already advanced.

This diagnostic runs **only Expert-controlled Train20**, not Validation,
not FINAL, and never trains or loads Control/Prestige Student models.
It does not rewrite the original Stage A FAILED gate.

### Windows

```powershell
git pull --ff-only
uv run pytest -q tests/test_prestige1_alignment_audit.py
uv run python -u scripts/run_prestige1_alignment_audit.py `
  --config configs/training/prestige1.toml
```

Output (new, refuse overwrite): `diagnostics/prestige1/stage_c_v22_train_alignment.json`

Each Train20 control frame compares:
- Legacy `target_labels`: next target with timestamp >= current timestamp
- Counterfactual `unresolved target`: first not used/missed by physical engine
- Whether legacy points to an already-resolved note, or skips an overdue
  not-yet-resolved note
- Mask and 200ms-near flags for both interpretations
- Timing discrepancy, chord-class discrepancy, teacher positive commands
- Short capped mismatch examples, per-anchor and aggregate counts
- Expert physical Hit/Miss/Too Early/Overload/KeyDown counts

The **counterfactual diagnostic label is not used for training**. In
particular, overdue unresolved events are masked rather than assigned
zero remaining time. Remaining chord counts only unresolved simultaneous
notes, which can differ from the original schedule's chord class.

The dataset Train20 anchor manifest must match saved Stage A v2 provenance.
Input SHA256 hashes are checked before/after, including previous Stage A,
Stage B, Stage C, pretrained GRU, core, and both checkpoints. Old Best is
verified; existing artifacts and FINAL are untouched. Validation20 is
not evaluated and is not used for model/hyperparameter selection.

**Interpretation:** A measurable label/physical-target mismatch motivates
subsequent Train-only experiments, but does not prove the mismatched label
alone caused any observed Stage C Overload, nor show that a corrected
label increases closed-loop Hits or SAFE.
