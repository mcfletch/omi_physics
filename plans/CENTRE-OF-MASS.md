# Honouring a body's centre of mass

*Proposed.*

`model.Motion` carries `centerOfMass`, `omi_gltf` reads and writes it, and
nothing in the solver looks at it. A body's origin is its mass, the centre of
its collider, and the transform its node is drawn at — one point doing three
jobs. A body whose weight is not in the middle of its shape cannot be
described.

## What that costs

A rigid body tips when the sideways pull on it passes `(track / 2) / h`, where
`h` is the height of its mass. With the mass pinned to the middle of the
collider, `h` is half the shape's height plus whatever the shape is held off
the ground — which for a car is most of the bodywork.

GLinting Steel's car is the worked example, in
[`glisteel/plans/CAR-ROLLS-OVER.md`](../../glisteel/plans/CAR-ROLLS-OVER.md).
Its mass sat 0.710 m up on a 1.58 m track: **1.11 g**, against tyres worth
1.9 g. It found the rollover before it found the limit of grip, so full lock on
flat ground put it on its roof from 72 km/h upward, with nothing to trip on.

The game's own fix is to hang the wheels and the bodywork higher on the body so
the body rides lower between them — which moves the collider down with the
mass, because they are the same point. That bought 1.11 g → 1.52 g and stopped
the rolling, and then ran into its own limit: the wheels have to stay *below*
the mass they carry, so the drop is capped at about 0.17 m and the tipping
point at about 1.5 g. Past that the body pivots on its springs on landing
instead of settling. An electric car's mass really is in the floor, well below
the middle of its bodywork; there is no way to say so.

It is not only vehicles. Anything loaded — a trailer with weight over its axle,
a crate packed on one side, a character carrying something — has the same
sentence to say and no way to say it.

## Two ways to spell it

The choice is which point the world's `position` array holds. Everything else
follows from it.

**A. `position` stays the node transform; the mass is an offset from it.**
Matches glTF, where the body's node is the transform and `centerOfMass` is
stated in its space. Every torque arm becomes `point - (position + R @ com)`,
and — the part that is easy to miss — a free body rotates about its *mass*, so
the integrator has to swing the origin around the centre rather than spinning
the body about the origin.

**B. `position` holds the mass; the collider is offset from it.**
Everything in the solver already treats `position` as the centre of mass, so
the dynamics need no change at all. The cost moves to collision: shapes acquire
a local offset, and broadphase bounds, narrowphase and raycasts all have to
apply it. The node transform becomes `position - R @ com`, derived on the way
out.

**B is the one to build.** The dynamics are the part that is subtle and easy to
get quietly wrong; the offset-shape work is mechanical and is a capability the
engine wants anyway, since glTF hangs a collider off a child node and this world
has no way to represent that today either. It also keeps `centerOfMass` honest
for round-tripping: it stays a property of the body rather than becoming a
correction applied in two places.

## What has to change

Under B:

- **`world.add_body`** — take `motion.centerOfMass`, store it per body, and
  place the simulated point at `position + R @ com` while remembering that the
  caller gave a node position.
- **Shapes get a local offset.** A collider is attached at `-com` in the body's
  frame. `broadphase` bounds, `narrowphase`/`collide`/`gjk` and `raycast` all
  read it.
- **Inertia.** `inertia_diagonal` returns a shape's inertia about *its own*
  centre. Offset from the mass, it needs the parallel-axis theorem, and the
  result is no longer diagonal in the body frame — either carry a full tensor
  or keep the diagonal and document the approximation. `_fill_mass` is where
  this lands, along with `inertiaDiagonal` when a file supplies one.
- **The places that measure a torque arm from a body's position**, which are
  the ones to be exhaustive about: `solver.py` contact arms (`c.point -
  world.position[a]`), `joints.py` anchors, and
  `vehicle.py`'s `_apply`/`_velocity_at`, which take their centre from
  `RaycastVehicle.position()`. Under B these are already right, and the job is
  to prove it rather than to change them.
- **`world.position` readers.** Anything drawing a body — including
  `Car.follow` in glisteel and every `position[i]` in the tests — wants the node
  point, not the mass. Give the world an explicit accessor for each and make
  the distinction impossible to confuse in a name.
- **The accelerators.** `_solver_native.pyx` and `_collide_native.pyx` mirror
  the Python; whatever changes shape or arm layout changes there too, and
  `docs/ACCELERATORS.md` describes the contract that keeps the two honest.
- **`omi_gltf`** already round-trips the field, and should keep the value it was
  given rather than one recovered from a shifted transform.

## What must not change

A body that leaves `centerOfMass` at the origin has to behave exactly as it
does today, to the bit where the arithmetic allows. That is the acceptance
test: the existing suite passes untouched, and a body with a zero offset
integrates identically to one built before the change.

Then the case this is for: a box whose mass is set below its middle stands
under a sideways push that tips the same box with a centred mass, and the
tipping point moves where the arithmetic says it should.

## What it would buy the car

The mass in the floor at about 0.35 m, the collider still wrapped around the
bodywork, and the wheels back where a car's are. `(1.70 / 2) / 0.35` is
**2.4 g** — beyond anything the tyres can ask for, so the limit becomes grip
and losing it becomes a slide. `CarSpec.mass_drop`, and the note that explains
why it is capped, would both go away.
