# omi_physics plans

Design notes and work not yet done. Deep-dive documentation for what *is*
lives in [docs/](../docs/README.md); this is what is intended.

| Plan | Status | Description |
|------|--------|-------------|
| [COLLISION-EVENTS.md](../../openglcontext/plans/COLLISION-EVENTS.md) | landed, one part open | Per-step contact events, phase 1 of OpenGLContext's collision-events plan. A `ContactTracker` after the solve records begin/persist/end per pair (sleeping pairs carried, not ended), a `ContactLog` holds them across the steps of one `advance()`, and `add_contact_listener` calls in-step. Bodies are named by a generation-checked `BodyRef`, so `remove_body`'s slot reuse no longer credits events to the wrong body. Also: native impulse write-back without warm start, `TriggerEvent` into the same log, `Trigger.collisionFilter` honoured, and a later pre-solve verdict hook. Landed with the pre-solve hook, a ray `filter=`, and kinematic bodies entering triggers; the character controller's proxy body is open. |
| [CENTRE-OF-MASS.md](CENTRE-OF-MASS.md) | proposed | Honour `Motion.centerOfMass`, which the data model carries and reads and writes through glTF but the solver ignores. A body's origin is currently its mass, its collider and its node transform all at once, so a body whose weight is not at the middle of its shape cannot be described — which is why a car with a low floor tips at 1.1 g and rolls over instead of sliding. Covers the two ways of spelling it, the places that measure a torque arm from a body's position, inertia under the parallel-axis theorem, the accelerators that mirror the solver, and what must not change for the bodies that leave it at the origin. |

## Review follow-ups

The workspace review of 2026-10-31 (origin/develop..develop) assigned these
findings to omi_physics. OpenGLContext's `plans/NATIVE-PHYSICS.md` holds the
other areas' follow-ups; omi_physics' are kept here because the engine no longer
depends on this project.

| Finding | Outcome |
|---------|---------|
| M19 | Fixed. A box across an edge between coplanar mesh triangles was pushed along the slope (20 degree cube moved 2.58 m). `TriangleMeshProxy.internal_edges` marks edges shared with a neighbour within 5 degrees of the same plane, and a sideways push across one is replaced by the face normal. The 20 degree cube moves 0.015 m in 300 steps; 0 of 60 random placements on 5 to 28 degree slopes slide and 0 of 72 in the review's probe tumble. |
| M20 | Fixed. The README, docs/README.md and the package description say that OpenGLContext uses pyjoltphysics and omi_physics is standalone and deprecated. |
| M21 | Fixed. A flat `convex` is bounded by its polygon in `Shape.hull_planes`, so rays meet it; `unsupported_shapes` names a `convex` on a line or at one point. |
| L25 | Fixed. `update`, `set_move`, `set_fly_move`, `apply_impulse` and `knockback` raise ValueError at the entry. |
| L26 | Fixed in omi_physics. `hull_volume` sums as OpenGLContext's copy does, so the copies agree bit for bit; `tests/test_hull_copies_agree.py` compares hulls, volumes, concavities and pieces over 201 hard clouds where OpenGLContext is installed. The duplication is noted in `hull.py` and docs/ARCHITECTURE.md. Left for OpenGLContext: its `cooking.py` docstring and license.txt do not say the code was relicensed by its author. |
| L27 | Fixed. A cone's side facets tilt by `(rb - rt) * cos(pi / sides)`, so they pass through the corners of both ends. |
| L29 | Fixed. `unsupported_shapes` no longer mentions a missing hull cast, and the dashes and "whole frame"/"hopeless" are out of the collide.py and character.py rows of docs/ARCHITECTURE.md. |
