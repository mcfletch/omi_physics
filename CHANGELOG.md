# Changelog

Notable changes to `omi_physics`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/); `0.x` makes no
compatibility promise.

## [0.4.0] - unreleased

### Changed

- `ThreadedSimulation.start()` is safe against concurrent calls. `with_world()` answers an
  object that wraps the lock; its block receives the world.
- A `cylinder` is a flat-ended cylinder rather than a capsule of its height.
- A triangle mesh against a triangle mesh makes no contacts.
- `SequentialImpulseSolver` no longer takes `position_iterations` or
  `baumgarte`, which nothing read.

- A box against a triangle of a mesh makes a contact at each corner that stands on the
  triangle's face (`box_face_contacts`), so a box rests on a static mesh on four points
  and friction holds it on a slope as it does on a box. A box over an edge or a
  triangle under the middle of a face keeps the single contact it had.
- A world stepped with a static body moved since the last step refits the boxes of the
  bodies that move as well as the moved body's. A body above a floor no longer falls through it.
- A crouched character stands up on a slope: the capsule tested for room is raised by
  `CharacterController.STAND_CLEARANCE` (2 cm).

- `GridSurface.lipschitz` is the steepest slope of the grid, and `sphere_contacts` leaves out the
  spheres too far above a grid to touch it and finds the few left one at a time with numbers
  (`GridSurface.sphere_contact`): a capsule standing on a grid ground costs 4 times less to collide.
  The slide down the tangent planes for arrays is `sphere_contacts_by_sliding`.
- A step reads only the static bodies recorded as added, removed or placed since the last one
  (`layout_epoch`): the box refit, the broad phase's index of inert bodies, the copy of previous
  poses, the sleep pass, the narrow phase's rotations and the solver's arrays no longer read or
  allocate a row for every body. A world of 100,000 static posts and a ball steps in 0.4 ms, from
  1.2 ms; the forest walker's frame is 2.2 ms, from 6.3 ms.
- A pose written into `position` or `orientation` of a static body is taken by a step only after
  `PhysicsWorld.moved(rows)`; `OMI_PHYSICS_CHECK_STATICS=1` (`PhysicsWorld.check_statics`) makes
  a step raise for one that was not recorded. A static sensor's placement and removal are recorded
  too.
- A dynamic body asleep against a static body that is placed, added or removed is woken.
- A capsule's box is derived from its pose as a box's is, and is not measured through a proxy
  each step. A body that is not turning keeps its orientation through the integration.
- A character holds the static bodies it gathered across updates while the query's
  `static_epoch()` holds (`WorldQuery.static_epoch`: the layout epoch, where no fixed body is
  kinematic), so a walk asks the static grid once in several updates.
- A character among height fields alone, with no dynamic body near, is collided with them with
  numbers (`collide.upright_capsule_heightfield_pushes`, `HeightSurface.column_contacts`): no
  capsule proxy, box test or contact object. On a grid, spheres higher than
  `heightfield.local_clearance` of the steepest slope within `LOCAL_REACH` radii are left out. The controller's
  share of a step on a hill grid is less than half what it was. Poses agree with the contact solver's to 1e-9 m a step;
  `CharacterController.ground_fast_path = False` turns it off.

### Fixed

- A fixed body's world-space proxy is dropped when its slot is emptied, so a body added into
  a slot a removed one held is collided with at its own shape and place, not the removed
  body's. A character on a streamed ground fell through the chunk that took a removed
  chunk's slot.

### Added

- `CharacterController` integrates gravity exactly, so a jump reaches
  `jumpHeight` and lands after `2 sqrt(2 h / g)` at every step size.
- `CharacterController.slide`, `knock_down`, `get_up`, `knockback`, `state`
  (`CharacterState`), `landing`, `standing_on` and `pushing`.
  `CharacterCapabilities` gains `mass`, `pushMass` and `pushSpeed`.
- `raycast` and `raycast_many` cast at convex hulls, by clipping the ray against
  the planes of the hull's faces. `unsupported_shapes` names none of the shapes a
  world holds.
- A world that adds, removes or places a few fixed bodies at a time, such as a
  streamed rock among 230,000 trunks, keeps the broad phase's index of its fixed
  bodies: up to `broadphase.LOOSE_LIMIT` changes since the build are swept with
  the movers, and a step after one costs the world's ordinary step and not a
  sort of every fixed body (about 50 ms at that size). A body added or placed
  with `add_body` or `place_body` arrives with its box fitted at its pose and
  is not refitted by the next step, and a body that takes a freed slot or leaves
  one changes its own row of the refit's shape table.
- `CollisionQuery.static_near(lo, hi)` lists the fixed bodies whose boxes overlap
  a box, from a grid over their footprints (`omi_physics.staticindex`) that
  `WorldQuery` shares between every query on a world. The grid reads up to 4096
  bodies added, removed or placed since it was built from the world
  (`PhysicsWorld.layout_epoch`, `layout_changes_since`) and is rebuilt past that or
  after `add_bodies`. `WorldQuery.dynamic_near` reads the boxes of the dynamic
  bodies only. `CharacterController` collides with the
  bodies near its capsule, so a step costs the same in a world of 200,000 posts as
  in one of 200. `PhysicsWorld.sensor` is the column of sensor flags.
- `CharacterController.slide(steer=True)`, `extend_slide` and `end_slide`: a slide the
  input still steers, one that goes on for a time from now, and one cut short. The
  capability `jumpFollowsSlope` keeps the vertical speed of a run down a slope in a
  jump.
- `omi_physics.collisionquery`: the `CollisionQuery` protocol and `WorldQuery`.
  `CharacterController` takes either a `PhysicsWorld` or a `CollisionQuery`, and
  treats dynamic bodies as obstacles, floors and pushable weights.
- `ThreadedSimulation.post(fn)` and `call(fn)`: a command inbox any thread
  writes to, run on the simulation thread at the start of the next tick.
  `call` answers a `concurrent.futures.Future`. `wait_steps(n, timeout)` waits
  for the tick count.
- `ThreadedSimulation` blocks while no dynamic body is awake, no kinematic
  body moves, the world holds no joint and the inbox is empty; `wake()` and
  `block_when_idle=False` control it. `with_world()` wakes it on leaving.
- `PhysicsWorld.add_step_hook(before, after)` and `omi_physics.Step`: hooks
  around every fixed step, inline or threaded.
- `PhysicsWorld.add_bodies(shape, positions, ...)`: many static bodies sharing
  one shape, filled by slice.
- A world of many static bodies steps in time that follows the bodies that
  move: the broad phase indexes static bodies with a collider and no trigger
  apart from the rest once there are 256 of them, boxes of static bodies are
  refitted only when a static body has moved, and the sleep pass, the force
  integration, the narrow phase and the contact solver gather the bodies they
  need instead of walking every row. The pairs reach the narrow phase in the
  order a sweep of every body gives them.
- `Shape.heightfield`: a static ground from a grid of heights (bilinear, exact
  cell normal) or from a function `sample(x, z)`, collided with spheres,
  capsules, cylinders, boxes and convex hulls and cast at by rays
  (`omi_physics.heightfield`). `Shape.hull_planes()` gives a convex shape's
  faces.
- Compound shapes: `Shape.compound` and `ShapePart`, shapes placed at offsets
  in one body's frame, collided, cast at and weighed part by part.
- Cylinders with flat ends, narrowing to a cone (`radiusTop` 0), collided,
  cast at and weighed as such.
- `massprops`: volume, centroid and inertia of every shape.
- Articulated joints: `BallJoint`, `HingeJoint`, `SliderJoint`,
  `UniversalJoint`, `TwoAxisHingeJoint` and `MotorJoint`, with stops, bounce,
  motors, a suspension and angle, rate and anchor readings, built from rows
  (`jointrows`) solved together and warm started, compiled where built.
  `remove_joint_constraint`. Jointed bodies do not collide unless
  `collide_connected` is set.
- `PhysicsWorld.set_pair_rule` and `model.ContactParameters`: which pairs
  collide, and with what restitution, bounce speed, friction in two
  directions, surface speed, slip, softness and error correction.
  `PhysicsWorld.contact_solver`; the solver's `correction` and `softness`.
- `PhysicsWorld.set_force`, `set_torque`, `set_exact_rotation` and
  `set_sleep`; `sleep_linear_speed`, `sleep_angular_speed`, `sleep_time`.
- Contact events on every step. The world records one `ContactEvent` per
  touching pair, with a phase of `'begin'`, `'persist'` or `'end'`, the closing
  speed, the normal and friction impulses, slip, depth, a point and a normal.
  Events go to contact listeners inside the step
  (`PhysicsWorld.add_contact_listener`) and into a bounded `ContactLog`
  (`PhysicsWorld.contact_log`) that a consumer drains. `contact_reporting` is
  `'off'` (the default), `'flagged'` or `'all'`; `report_persist` adds
  `'persist'` and `'stay'` events. See the README's "Being told when things
  touch".
- `PhysicsWorld.log_events`: off, events go only to the listeners and the log
  stays empty.
- `BodyRef(index, generation)`, `PhysicsWorld.ref`, `alive` and `handle_of`: a
  reference to a removed body never matches the body put in its slot.
- `PhysicsWorld.set_contact_filter`: a `PairPreview` of a reported pair before
  its first solve, answered with a `Verdict` of `SOLVE`, `IGNORE_STEP` or
  `IGNORE_PAIR`.
- Trigger events (`TriggerEvent`, with the entering body's `point`) join the
  contact log in step order. A trigger's `collisionFilter` is read from glTF,
  written back and honoured, and a kinematic body enters triggers.
- Sensor bodies (`add_body(..., sensor=True)`, `PhysicsWorld.is_sensor`): they
  enter triggers and report touches with `solved=False`, and nothing is solved
  against them. A `CharacterController` built with `body=True` has one, and
  reports the static colliders it begins and stops touching through
  `PhysicsWorld.report_contact_events`.
- A `filter` keyword on `raycast`, `raycast_many` and `bodies_along`: a
  `CollisionFilter` for the ray itself.
- `ThreadedSimulation.latest_and_events()` returns the newest snapshot with
  every event published up to it, and `flush_events()` publishes what a
  removal made under `with_world()` logged without waiting for a tick.
- `ThreadedSimulation.with_world()` is re-entrant, so code holding it may call
  something that takes it again.
- `kinematic.KinematicMover`: a kinematic body sent to a target position at
  a set speed, placed exactly on it when it arrives.
- `gravity.Region` is a protocol: any object with `contains(position)` is a
  region.
- `PhysicsWorld.reports_pair(a, b)`: whether `contact_reporting` records the
  pair of body indices `a` and `b`.
- `backend.has_gl_compute()`: whether the current GL context can dispatch
  compute shaders, the test `select_backend('auto')` makes.

### Changed

- `remove_body` ends the removed body's contacts with `reason='removed'`,
  exits the triggers it was in at once, and forgets its warm start.
- A contact or trigger listener that raises no longer stops the step part-way.
  The other listeners are still called, the step or removal finishes, and the
  first exception is raised from `step()` or `remove_body()` once it has.
- `raycast_many` tests every ray of a bundle against the shared candidate
  triangles in one pass.
- A pair asleep against a static body stays touching rather than ending, and a
  pair with no dynamic body is not solved.
- Where GL 4.3 compute is present but the GPU backend fails to build,
  `select_backend('auto')` and a world crossing its `gpu_threshold` log a
  warning with the traceback on the `omi_physics.backend` logger, then use
  numpy. The compute probe answers "no compute" for a missing PyOpenGL, a GL
  error or an unreadable version string; any other exception propagates.

### Fixed

- `cook_shape(..., strategy='primitive')` fitted every box as a sphere, because
  the eight corners of a box are all one distance from its centre. A sphere is
  now chosen only where the points also reach that distance along each axis.
- EPA expands the face the origin lies on, so a capsule resting on a box is
  found.
- GJK reports an overlap when the origin lies on an edge or a face of its
  tetrahedron to within rounding, rather than searching round that edge until
  it runs out of iterations.

## [0.3.2]

The last release before this changelog.
