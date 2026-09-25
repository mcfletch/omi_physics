# Changelog

Notable changes to `omi_physics`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/); `0.x` makes no
compatibility promise.

## [0.4.0] - unreleased

### Added

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
  contact log in step order. A
  trigger's `collisionFilter` is read from glTF, written back and honoured, and
  a kinematic body enters triggers.
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
- `gravity.Region` is a protocol: any object with `contains(position)` is a
  region.

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

### Fixed

- EPA expands the face the origin lies on, so a capsule resting on a box is
  found.
- GJK reports an overlap when the origin lies on an edge or a face of its
  tetrahedron to within rounding, rather than searching round that edge until
  it runs out of iterations.

## [0.3.2]

The last release before this changelog.
