# The step pipeline

`PhysicsWorld.step(dt)` advances the whole world by one **fixed** timestep. A
real-time caller drives it through `advance(real_dt)`, which runs an accumulator
so the simulation always steps in equal `dt` slices regardless of frame rate.

```mermaid
flowchart TB
    subgraph advance["world.advance(real_dt)"]
        acc[accumulate real_dt] --> loop{accumulator ≥ fixed_dt?}
        loop -->|yes| step[step&#40;fixed_dt&#41;]
        step --> loop
        loop -->|no| interp[return interpolation alpha]
    end
```

## One step, stage by stage

```mermaid
sequenceDiagram
    autonumber
    participant W as PhysicsWorld
    participant B as backend
    participant BP as broadphase
    participant NP as narrowphase
    participant SV as solver
    participant JO as joints
    participant SL as sleeping

    W->>W: save prev_position / prev_orientation
    W->>B: integrate_forces(dt)
    Note right of B: v += g·dt, then damping and drag
    alt a collider exists
        W->>B: refit_aabbs()
        W->>BP: pairs()  → candidate overlaps
        BP->>NP: candidate pairs
        NP->>NP: generate contacts (SAT / GJK)
        NP->>SV: contact manifolds
        SV->>SV: warm-start + velocity iterations
        SV->>SV: position (penetration) correction
    end
    opt joints present
        W->>JO: solve_joints(dt)
    end
    W->>B: integrate_positions(dt)
    Note right of B: x += v·dt, then q ← integrate(q, ω, dt)
    opt sleeping enabled
        W->>SL: update_sleep(dt)
    end
    W->>W: time += dt
```

## What flows between stages

| Stage | Reads | Writes |
| --- | --- | --- |
| **integrate forces** | `linear/angular_velocity`, `gravity`, damping, drag | `linear/angular_velocity` |
| **refit AABBs** | `position`, `orientation`, shape half-extents | `aabb_min`, `aabb_max` |
| **broadphase** | `aabb_min/max`, collision filters | candidate `(i, j)` pairs |
| **narrowphase** | candidate pairs, `position`, `orientation`, shapes | `Contact` manifolds (normal, points, depth) |
| **solver** | contacts, velocities, inverse mass/inertia, materials | `linear/angular_velocity`, positional correction, `Contact.approach` |
| **joints** | joint definitions, body poses/velocities | velocities, poses |
| **integrate positions** | velocities | `position`, `orientation` |
| **sleeping** | velocities over time | `awake`, `sleep_timer` |

## Broadphase → narrowphase → solve

```mermaid
flowchart LR
    subgraph broad["Broadphase — cheap, approximate"]
        SAP[sort AABBs on an axis<br/>sweep &amp; prune] --> CAND[candidate pairs<br/>&#40;may overlap&#41;]
    end
    subgraph narrow["Narrowphase — exact"]
        CAND --> ROUTE{shape pair}
        ROUTE -->|box·box / sphere| FAST[SAT / analytic<br/>collide.py · _collide_native]
        ROUTE -->|general convex| GJK[GJK / EPA<br/>gjk.py]
        FAST --> MAN[contact manifolds]
        GJK --> MAN
    end
    subgraph solve["Solver — sequential impulse"]
        MAN --> ISL[build islands<br/>connected bodies]
        ISL --> WS[warm start from<br/>last step's impulses]
        WS --> VI[velocity iterations<br/>Gauss–Seidel]
        VI --> PC[position correction]
    end
```

**Broadphase** never claims two bodies *are* touching — only that their AABBs
overlap, so they are worth an exact test. It exists to keep the exact test off the
O(n²) all-pairs cost.

**Narrowphase** turns each surviving pair into a contact manifold: a normal, one
or more contact points, and a penetration depth. Box-box and sphere pairs — the
overwhelming majority in a typical scene — take a vectorized SAT / analytic path
(and the `_collide_native` accelerator when present); general convex shapes go
through GJK/EPA.

**Solver** groups contacting bodies into independent *islands* and solves each
with sequential-impulse Gauss–Seidel: it iterates over the contacts applying
velocity impulses until they stop interpenetrating, warm-starting from the
previous step's impulses so stacks settle quickly. This is the stage the
`_solver_native` accelerator replaces.

After the velocity iterations comes a **split-impulse position pass**, which
moves overlapping bodies apart along the contact normal without giving them the
velocity that movement would imply — so a stack settling does not gain the
energy a Baumgarte bias would put into it. Two things bound what it does:

- Each contact is corrected against the movement the pass has **already**
  applied, not against the overlap the narrow phase measured. A box resting on a
  floor meets it at four points reporting the same overlap, and four full
  corrections would push it four times as far as it is in.
- `SequentialImpulseSolver.max_correction` (metres, default `0.2`) caps how far
  one step may push a pair apart, so a body that arrives deep inside another —
  placed there, or driven there at speed — climbs out over a few frames rather
  than being flung across the level in one.

Overlap below `slop` (metres, default `0.005`) is left alone: contact is
discrete, and a resting stack that is corrected to exactly zero overlap
separates, falls and lands again every step.

**Contacts against a static body are solved last.** Both passes are
Gauss-Seidel, so what they correct last is what ends up satisfied, and where a
body is held between two contacts asking opposite things only one of them can
be. It has to be the static one: a body left overlapping another *dynamic* body
is pushed apart over the next few steps, while a body left inside the level's
own geometry is thrown out of the far side of it as soon as its centre passes
the middle — past that point the nearest separating axis is the wrong one, and
the ground ejects it downwards. A tenth of a gramme under 243 kilogrammes is
enough to reach it.

**Where the mass ratios reach.** Sequential impulses pass a load down a stack
one contact at a time, so what a stack holds depends on how many iterations the
load has to travel through. `velocity_iterations` (default `10`) settles a stack
of boxes of comparable weight; a light body pinned between two much heavier ones
is being pushed by a force those iterations cannot balance, and it is squeezed
out sideways rather than held. Somewhere past a hundred to one, in a stack
several bodies deep, that is what happens. More iterations buy more ratio, at
their own cost per step.

Before any of that it writes each contact's **approach** — how fast the pair
were closing along the contact normal when they met. Restitution is scaled by
it, and `PhysicsWorld.impact_on` answers from it. It has to be taken here and
nowhere later: the velocity iterations exist to remove exactly that velocity, so
a game reading the same quantity off the bodies after the step gets a number
that says a square-on impact never happened.

## Determinism

On the `NumpyBackend` the pipeline is float64 and every stage visits bodies and
contacts in a stable order, so a scene replays bit-for-bit. Two things break that
deliberately:

- The **GPU backend** (`glcompute.py`) computes integration in float32, so its
  trajectories match the CPU path only within tolerance.
- **Sleeping** is timing-dependent only in the sense of accumulated `dt`; with a
  fixed `dt` it too is deterministic.
