# omi_physics plans

Design notes and work not yet done. Deep-dive documentation for what *is*
lives in [docs/](../docs/README.md); this is what is intended.

| Plan | Status | Description |
|------|--------|-------------|
| [CENTRE-OF-MASS.md](CENTRE-OF-MASS.md) | proposed | Honour `Motion.centerOfMass`, which the data model carries and reads and writes through glTF but the solver ignores. A body's origin is currently its mass, its collider and its node transform all at once, so a body whose weight is not at the middle of its shape cannot be described — which is why a car with a low floor tips at 1.1 g and rolls over instead of sliding. Covers the two ways of spelling it, the places that measure a torque arm from a body's position, inertia under the parallel-axis theorem, the accelerators that mirror the solver, and what must not change for the bodies that leave it at the origin. |
