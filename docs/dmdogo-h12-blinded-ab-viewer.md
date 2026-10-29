# DMDOGO H1.2 — Offline blinded A/B fingering viewer

## Scientific status

H1.1 reconstructed 22 independent full-chart counterfactual assignments on three previously selected Train charts (10,231 events). Seven have strictly lower **frozen six-term heuristic cost**, not verified human preference or alternate physical Motor success.

H1.2 builds an **offline review interface**, not human evaluation results. No human judgments are created automatically, no physical Motor or timing process runs, and no Validation20/FINAL data is opened.

## Create-once outputs: share only the HTML

- Public standalone viewer: artifacts/dmdogo-h12/blind-review-r1.html
- **PRIVATE** unblinding key: reports/dmdogo-h12-private-key-r1.json

Outputs must be in separate directories, and the builder refuses to overwrite either one. **Never share the private JSON with a reviewer.** The shareable HTML contains no original/alternate identity, source candidate ID, chart SHA, six-term score, cost direction or random seed. It has all 22 independently randomized candidate codes.

The original appears as A in exactly 11 cases and as B in 11 cases. Balance is enforced within each source chart: 5/5, 5/5, 1/1. The private key contains the source H1.1 SHA256, 256-bit seed, blinded case code, source candidate ID, original side and actual heuristic difference.

## Viewer behavior

- One **offline self-contained HTML file** with embedded, blinded event data; no CDN, network, server, or local-file fetch is needed.
- Every case shows exact shared source event IDs, due timestamps, and A/B axis assignments for the complete chart.
- Filter modes: only events with changed assignments, ±8-event context, or all events.
- Pagination displays 80 rows at a time, including for changes spanning 2,906 events.
- Highlighted rows indicate differing abstract axes, not anatomical finger movement or real playability.
- Reviewer can enter qualitative preferences (A, B, tie, uncertain), a separate ease-of-play impression, confidence, and optional notes.
- Annotations are **not sent or saved automatically**. Click Export to save user-entered results to local JSON. Blank fields remain null.
- Export includes packet_id and anonymous case codes to join with the separate key only after review.

This is visual review of abstract key assignments. It does not show audio, actual physical keypress trajectories, or human fingering ground truth. Static preference judgments cannot establish Motor success.

## Strict H1.1 source checks

Builder verifies exactly 3 charts, 10/10/2 alternative assignments, 22 distinct IDs, 10,231 original events, 7 lower-objective paths, complete ordered A/B axes and chords, all saved changed-event values, legitimate frozen eight-axis vocabulary, and explicit absence of alternate Motor/human observations. H1.1 has already performed deeper H0.6 replay and provenance checks. The key additionally pins the H1.1 source SHA256.

All 22 counterfactual worlds remain independent. Never combine alternate paths into one fictitious physical episode.

## Windows reproduction

Stay on develop/dmdogo. Preserve local configs/training/prestige1.toml and all untracked artifacts; use only a fast-forward pull, not reset/rebase/stash/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h12_blind_review.py

uv run python scripts/build_dmdogo_h12_blind_review.py `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --viewer-output artifacts/dmdogo-h12/blind-review-r1.html `
  --private-key-output reports/dmdogo-h12-private-key-r1.json
~~~

Open artifacts/dmdogo-h12/blind-review-r1.html in a browser. There is no web server and no external dependency.

For a different review run, use new output filenames for **both** files. The random seed generates a new mapping and packet ID. Do not combine exported reviews and keys from different packet IDs.

## Next evidence boundary

After genuinely independent human annotations, a separate future analysis may match them to the private key and quantify observed judgments with case-level missingness and order balance. This phase does not obtain or fabricate those judgments, claim formal double-blind validity, or validate physical feasibility. Future Motor A/B validation requires a separately authorized experimental protocol.
