# omi_physics plans

Design notes and work not yet done. Deep-dive documentation for what *is*
lives in [docs/](../docs/README.md); this is what is intended.

| Plan | Status | Description |
|------|--------|-------------|
| [COLLISION-EVENTS.md](../../openglcontext/plans/COLLISION-EVENTS.md) | proposed | Per-step contact events, phase 1 of OpenGLContext's collision-events plan. A `ContactTracker` after the solve records begin/persist/end per pair (sleeping pairs carried, not ended), a `ContactLog` holds them across the steps of one `advance()`, and `add_contact_listener` calls in-step. Bodies are named by a generation-checked `BodyRef`, so `remove_body`'s slot reuse no longer credits events to the wrong body. Also: native impulse write-back without warm start, `TriggerEvent` into the same log, `Trigger.collisionFilter` honoured, and a later pre-solve verdict hook. |
| [CENTRE-OF-MASS.md](CENTRE-OF-MASS.md) | proposed | Honour `Motion.centerOfMass`, which the data model carries and reads and writes through glTF but the solver ignores. A body's origin is currently its mass, its collider and its node transform all at once, so a body whose weight is not at the middle of its shape cannot be described — which is why a car with a low floor tips at 1.1 g and rolls over instead of sliding. Covers the two ways of spelling it, the places that measure a torque arm from a body's position, inertia under the parallel-axis theorem, the accelerators that mirror the solver, and what must not change for the bodies that leave it at the origin. |
