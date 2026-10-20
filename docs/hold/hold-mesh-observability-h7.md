# Hold mesh observability (H7)

## Audited CPU mesh rule

`scrHoldRenderer.CreateMesh()` returns immediately for a Midspin start floor.
Otherwise it computes a shader angle, samples 100 longitudinal intervals per
full revolution, and allocates two strip vertices per interval boundary:

```text
segments = floor(100 * shaderAngle / (2*pi))
vertices = 2 + 2*segments
triangle indices = 6*segments
```

Each positive `holdLength` adds one full revolution and therefore exactly 100
segments.  In the fixed 180-degree fixture, duration 0/1/2/3 produces
50/150/250/350 segments and 102/302/502/702 vertices.

This makes duration identifiable from the generated Mesh buffer when the base
geometry is known.  Mesh buffer length is internal renderer state, not a
player-visible observation, and is prohibited as a Student feature.

The audited source sets `_EntryAngle`, `_AngleLength`, `_StartPosition`,
`_EndPosition`, widths, radii, direction, UVs, and completion on a material.
The corresponding Unity vertex/fragment shader source is not present in the
audited C# snapshot.  Game assets contain the property names but did not provide
an audited shader formula.  Consequently the exact post-shader curve and pixel
appearance cannot be reconstructed from this source alone.

## Counterfactual pixel probes

The implementation separates two explicitly labelled shader hypotheses:

1. `OPAQUE_CONSTANT_RADIUS`: repeated revolutions overlap the same pixels.
   Positive durations are ambiguous despite different Mesh vertex counts.
2. `PROXY_RADIAL_DRIFT`: revolutions separate radially.  Clean synthetic
   pixels distinguish duration 0/1/2/3 across tested zoom and rotation.

The pixel sensor receives only a frame-sized set of lit pixels and causal
timestamps.  It receives no duration, angle length, vertex count, or target
time.  It returns a candidate set; multiple exact templates or occlusion cause
ambiguity/abstention.

The radial-drift result is a sensor prototype, not Stock evidence.  The opaque
counterexample proves that CPU vertex-count identifiability does not imply
pixel identifiability.

## Camera, overlap, and timing

- Camera rotation and uniform zoom preserve the clean proxy distinction when
  calibrated from visible floor geometry.
- Exact self-overlap removes revolution-count information from an opaque mask.
- Cropping/occlusion invalidates exact matching and causes abstention.
- Midspin has no Hold mesh in the audited `CreateMesh` path.
- Skin texture, alpha blending, render ordering, nonuniform camera transforms,
  and competing decorations remain live-renderer dependent.
- No live capture was supplied, so first-visible time, capture latency, and
  whether a unique duration arrives 44ms before release are unverified.

## H7 decision

`CODE_GENERATED_MESH_IDENTIFIABLE_PIXEL_OBSERVABILITY_UNVERIFIED`

H8 may accept a unique, causally timestamped pixel result only under an
explicit verified-renderer contract.  The current proxy must never activate a
production HUD-only release forecast.
