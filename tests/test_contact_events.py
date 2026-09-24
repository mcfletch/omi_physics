"""Per-step contact events: what touched what, on every step (no GL).

A game asks "did the box hit the floor" once a frame, and a frame is several
steps. The record here is taken on every step, so a bounce that begins and ends
between two frames is still reported, and it is reported once per pair rather
than once per manifold point.

Every test runs on both solver paths: the compiled kernel, which hands the
tracker its contact arrays, and the pure-Python solver, which does not.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from omi_physics import contactevents, model, omi_gltf, solver
from omi_physics.contactevents import (
    BodyRef,
    ContactBatch,
    ContactEvent,
    ContactLog,
    TriggerEvent,
    Verdict,
)
from omi_physics.threaded import ThreadedSimulation
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0


@pytest.fixture(params=['native', 'python'])
def solver_path(request, monkeypatch):
    """Run the test on the compiled solver, then again on the Python one."""
    if request.param == 'python':
        monkeypatch.setattr(solver, '_native', None)
    elif solver._native is None:
        pytest.skip('accelerators are not built')
    return request.param


def world_with_floor(restitution=0.0, sleep=False, **options):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         fixed_dt=STEP, sleep_enabled=sleep, **options)
    material = world.add_material(model.Material(restitution=restitution,
                                                 restitutionCombine=model.MAXIMUM))
    ground = world.add_shape(model.Shape.box((40, 1, 40)))
    floor = world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=ground, physicsMaterial=material),
                           position=(0, -0.5, 0))
    return world, material, floor


def drop(world, material, shape, height=2.0, x=0.0, **kwargs):
    return world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=world.add_shape(shape),
                                                  physicsMaterial=material),
                          position=(x, height, 0), **kwargs)


def begins(events, body=None):
    return [e for e in events if isinstance(e, ContactEvent) and e.phase == 'begin'
            and (body is None or body in (e.a.index, e.b.index))]


def bounce_count(frame):
    """The bounces a listener hears from a ball advanced in frames of ``frame`` s."""
    world, material, _floor = world_with_floor(restitution=0.6)
    ball = drop(world, material, model.Shape.sphere(0.2))
    heard = []
    world.add_contact_listener(heard.append)
    for _ in range(round(3.0 / frame)):
        world.advance(frame)
    return len(begins(heard, ball))


class TestEveryStepIsRecorded:
    def test_a_slow_frame_rate_hears_every_bounce(self, solver_path) -> None:
        at_120 = bounce_count(1.0 / 120.0)
        assert at_120 >= 3
        assert bounce_count(1.0 / 30.0) == at_120
        assert bounce_count(1.0 / 20.0) == at_120

    def test_the_log_holds_every_step_of_a_frame(self, solver_path) -> None:
        world, material, _floor = world_with_floor(restitution=0.6)
        ball = drop(world, material, model.Shape.sphere(0.2))
        world.contact_reporting = 'all'
        logged = []
        for _ in range(90):
            world.advance(1.0 / 30.0)
            logged += world.contact_log.drain()
        assert len(begins(logged, ball)) == bounce_count(1.0 / 120.0)
        steps = [e.step for e in logged]
        assert steps == sorted(steps)


class TestOneEventPerPair:
    def test_a_box_landing_flat_is_one_begin(self, solver_path) -> None:
        world, material, floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.8)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(60):
            world.step(STEP)
        landing = begins(heard, box)
        assert len(landing) == 1
        event = landing[0]
        assert event.points == 4
        assert (event.a.index, event.b.index) == (floor, box)

    def test_the_normal_points_from_a_to_b(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.8)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(60):
            world.step(STEP)
        event = begins(heard, box)[0]
        assert event.a.index < event.b.index
        assert np.allclose(event.normal, (0, 1, 0), atol=1e-6)
        assert event.point[1] == pytest.approx(0.0, abs=0.02)

    def test_a_blow_says_how_hard(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        ball = drop(world, material, model.Shape.sphere(0.2), height=0.2 + 1.25)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(90):
            world.step(STEP)
        event = begins(heard, ball)[0]
        falling = (2 * 9.81 * 1.25) ** 0.5
        assert event.approach == pytest.approx(falling, rel=0.05)
        # A 1 kg ball stopped dead: the impulse is its momentum.
        assert event.impulse == pytest.approx(falling, rel=0.1)
        assert event.depth >= 0.0


class TestPhases:
    def test_persist_is_reported_only_when_asked_for(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.51)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(10):
            world.step(STEP)
        assert [e.phase for e in heard] == ['begin']
        world.report_persist = True
        for _ in range(10):
            world.step(STEP)
        assert [e.phase for e in heard[1:]] == ['persist'] * 10
        assert all(box in (e.a.index, e.b.index) for e in heard)

    def test_leaving_the_floor_ends_the_pair(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.51)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(10):
            world.step(STEP)
        world.linear_velocity[box] = (0, 6, 0)
        for _ in range(10):
            world.step(STEP)
        phases = [e.phase for e in heard]
        assert phases == ['begin', 'end']
        assert heard[1].reason == 'separated'
        assert heard[1].impulse == 0.0

    def test_a_crate_that_falls_asleep_has_not_left_the_floor(self, solver_path) -> None:
        world, material, _floor = world_with_floor(sleep=True)
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.6)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(240):
            world.step(STEP)
        assert not world.awake[box]
        assert [e.phase for e in heard] == ['begin']
        world.linear_velocity[box] = (0, 6, 0)
        world.wake(box)
        for _ in range(10):
            world.step(STEP)
        assert [e.phase for e in heard] == ['begin', 'end']


class TestReportingLevels:
    def test_a_world_nobody_listens_to_records_nothing(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        drop(world, material, model.Shape.box((1, 1, 1)), height=0.8)
        assert world.contact_reporting == 'off'
        for _ in range(60):
            world.step(STEP)
        assert len(world.contact_log) == 0

    def test_flagged_reports_only_the_pairs_of_flagged_bodies(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        quiet = drop(world, material, model.Shape.box((1, 1, 1)), height=0.8, x=-3)
        loud = drop(world, material, model.Shape.box((1, 1, 1)), height=0.8, x=3,
                    report_contacts=True)
        world.contact_reporting = 'flagged'
        for _ in range(60):
            world.step(STEP)
        logged = world.contact_log.drain()
        assert begins(logged, loud)
        assert not begins(logged, quiet)
        world.report_contacts(quiet)
        world.linear_velocity[quiet] = (0, 3, 0)
        for _ in range(120):
            world.step(STEP)
        assert begins(world.contact_log.drain(), quiet)

    def test_a_listener_switches_reporting_on(self, solver_path) -> None:
        world, _material, _floor = world_with_floor()
        world.add_contact_listener(lambda event: None)
        assert world.contact_reporting == 'all'

    def test_a_listener_can_be_taken_away(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        drop(world, material, model.Shape.sphere(0.2), height=0.3)
        heard = []
        world.add_contact_listener(heard.append)
        world.remove_contact_listener(heard.append)
        for _ in range(30):
            world.step(STEP)
        assert heard == []


class TestTheLog:
    def test_it_is_bounded_and_counts_what_it_dropped(self) -> None:
        log = ContactLog(maxlen=3)
        log.extend(range(5))
        assert len(log) == 3
        assert log.dropped == 2
        assert log.drain() == [2, 3, 4]
        assert len(log) == 0

    def test_it_can_be_emptied(self) -> None:
        log = ContactLog()
        log.extend(range(5))
        log.clear()
        assert log.drain() == []


class TestSwitching:
    def test_an_unknown_level_is_refused(self) -> None:
        world, _material, _floor = world_with_floor()
        with pytest.raises(ValueError, match='contact_reporting'):
            world.contact_reporting = 'some'

    def test_switching_off_forgets_what_was_touching(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.5)
        world.contact_reporting = 'all'
        world.step(STEP)
        world.contact_reporting = 'off'
        world.step(STEP)
        world.contact_reporting = 'all'
        world.step(STEP)
        assert [e.phase for e in world.contact_log.drain()] == ['begin', 'begin']
        assert world.alive(world.ref(box))

    def test_removing_an_untouched_body_ends_nothing(self) -> None:
        world, material, _floor = world_with_floor()
        ball = drop(world, material, model.Shape.sphere(0.2), height=5.0)
        heard = []
        world.add_contact_listener(heard.append)
        world.step(STEP)
        world.remove_body(ball)
        assert heard == []

    def test_a_listener_that_raises_stops_the_step_and_nothing_is_left_queued(
            self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        drop(world, material, model.Shape.sphere(0.2), height=0.3)
        drop(world, material, model.Shape.sphere(0.2), height=0.3, x=2.0)

        def broken(event):
            raise RuntimeError('listener failed')

        world.add_contact_listener(broken)
        with pytest.raises(RuntimeError, match='listener failed'):
            for _ in range(60):
                world.step(STEP)
        world.remove_contact_listener(broken)
        assert not world._undelivered


class TestBatches:
    def test_joining_two_batches_keeps_both(self) -> None:
        from omi_physics.collide import Contact
        contact = Contact(0, 1, np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.01)
        solved = ContactBatch.from_contacts([contact])
        unsolved = ContactBatch.from_contacts([contact], solved=False)
        both = solved.concatenate(unsolved)
        assert both.solved.tolist() == [True, False]
        assert ContactBatch.empty().concatenate(solved) is solved
        assert solved.concatenate(ContactBatch.empty()) is solved


class TestImpulsesAreFilled:
    def test_without_warm_starting(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        world.contact_reporting = 'all'
        drop(world, material, model.Shape.sphere(0.2), height=0.6)
        world._collision.solver.warm_start = False
        for _ in range(60):
            world.step(STEP)
            if world.contacts:
                break
        assert world.contacts
        assert any(c.normal_impulse > 0.0 for c in world.contacts)

    def test_sliding_reports_friction_and_slip(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.5)
        world.report_persist = True
        heard = []
        world.add_contact_listener(heard.append)
        world.linear_velocity[box] = (4, 0, 0)
        for _ in range(3):
            world.step(STEP)
        sliding = [e for e in heard if e.phase in ('begin', 'persist')]
        assert sliding[-1].slip > 1.0
        assert sliding[-1].friction_impulse > 0.0


class TestBodyReferences:
    def test_a_reference_outlives_its_body(self) -> None:
        world, material, _floor = world_with_floor()
        first = drop(world, material, model.Shape.sphere(0.2))
        ref = world.ref(first)
        assert ref == BodyRef(first, world.ref(first).generation)
        assert world.alive(ref)
        world.remove_body(first)
        assert not world.alive(ref)
        second = drop(world, material, model.Shape.sphere(0.2))
        assert second == first
        assert world.ref(second) != ref
        assert world.handle_of(ref) is None

    def test_handle_of_answers_the_handle(self) -> None:
        world, material, _floor = world_with_floor()
        body = drop(world, material, model.Shape.sphere(0.2), handle='crate')
        assert world.handle_of(world.ref(body)) == 'crate'

    def test_removing_a_touching_body_ends_its_pairs(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        box = drop(world, material, model.Shape.box((1, 1, 1)), height=0.51)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(10):
            world.step(STEP)
        ref = world.ref(box)
        world.remove_body(box)
        assert heard[-1].phase == 'end'
        assert heard[-1].reason == 'removed'
        assert heard[-1].b == ref
        # The slot's next tenant starts from nothing.
        again = drop(world, material, model.Shape.box((1, 1, 1)), height=0.51)
        assert again == box
        for _ in range(10):
            world.step(STEP)
        assert [e.phase for e in heard] == ['begin', 'end', 'begin']
        assert heard[-1].b == world.ref(again) != ref

    def test_a_listener_may_remove_what_hit_something(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        shot = drop(world, material, model.Shape.sphere(0.1), height=0.5)
        heard = []

        def on_contact(event):
            heard.append(event)
            if event.phase == 'begin' and world.alive(event.b):
                world.remove_body(event.b.index)

        world.add_contact_listener(on_contact)
        for _ in range(60):
            world.step(STEP)
        assert [(e.phase, e.reason) for e in heard] == [('begin', None), ('end', 'removed')]
        assert shot in world._free


def trigger_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=STEP)
    zone = world.add_shape(model.Shape.box((2, 2, 2)))
    pad = world.add_body(model.Motion(type=model.STATIC),
                         trigger=model.Trigger(shape=zone), position=(0, 0, 0))
    return world, pad


def ball_in(world, position=(0, 0, 0), filt=-1):
    shape = world.add_shape(model.Shape.sphere(0.2))
    return world.add_body(model.Motion(type=model.DYNAMIC),
                          collider=model.Collider(shape=shape, collisionFilter=filt),
                          position=position)


class TestTriggers:
    def test_trigger_events_join_the_log(self) -> None:
        world, pad = trigger_world()
        ball = ball_in(world)
        world.contact_reporting = 'all'
        for _ in range(3):
            world.step(STEP)
        logged = world.contact_log.drain()
        assert logged == [TriggerEvent('enter', world.ref(pad), world.ref(ball), 0, 0.0)]

    def test_stay_is_logged_only_when_persist_is_asked_for(self) -> None:
        world, _pad = trigger_world()
        ball_in(world)
        world.contact_reporting = 'all'
        world.report_persist = True
        for _ in range(3):
            world.step(STEP)
        assert [e.phase for e in world.contact_log.drain()] == ['enter', 'stay', 'stay']

    def test_removing_a_body_inside_a_trigger_exits_it_once(self) -> None:
        world, pad = trigger_world()
        ball = ball_in(world)
        heard = []
        world.add_trigger_listener(lambda kind, t, o: heard.append((kind, t, o)))
        world.step(STEP)
        world.remove_body(ball)
        assert heard == [('enter', pad, ball), ('exit', pad, ball)]
        # The body that takes the slot enters afresh rather than staying.
        again = ball_in(world)
        assert again == ball
        world.step(STEP)
        assert heard[-1] == ('enter', pad, again)

    def test_a_kinematic_body_enters_and_leaves_a_trigger(self) -> None:
        world, pad = trigger_world()
        shape = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
        lift = world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(4, 0, 0)),
                              collider=model.Collider(shape=shape), position=(-3, 0, 0))
        heard = []
        world.add_trigger_listener(lambda kind, t, o: heard.append((kind, t, o)))
        for _ in range(240):
            world.step(STEP)
        assert [(kind, t, o) for kind, t, o in heard if kind != 'stay'] == [
            ('enter', pad, lift), ('exit', pad, lift)]

    def test_a_kinematic_body_makes_no_contact_with_a_static_one(self) -> None:
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=STEP)
        shape = world.add_shape(model.Shape.box((1, 1, 1)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape),
                       trigger=model.Trigger(shape=shape))
        world.add_body(model.Motion(type=model.KINEMATIC),
                       collider=model.Collider(shape=shape), position=(0.5, 0, 0))
        world.step(STEP)
        assert world.contacts == []

    def test_a_trigger_honours_its_collision_filter(self) -> None:
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=STEP)
        players = world.add_filter(model.CollisionFilter(collisionSystems=('player',)))
        crates = world.add_filter(model.CollisionFilter(collisionSystems=('crate',)))
        for_players = world.add_filter(model.CollisionFilter(
            collisionSystems=('sensor',), collideWithSystems=('player',)))
        zone = world.add_shape(model.Shape.box((4, 2, 2)))
        world.add_body(model.Motion(type=model.STATIC),
                       trigger=model.Trigger(shape=zone, collisionFilter=for_players))
        player = ball_in(world, (-1, 0, 0), players)
        ball_in(world, (1, 0, 0), crates)
        heard = []
        world.add_trigger_listener(lambda kind, t, o: heard.append((kind, o)))
        world.step(STEP)
        assert heard == [('enter', player)]

    def test_gltf_reads_and_writes_a_trigger_filter(self) -> None:
        doc = {'extensionsUsed': ['OMI_physics_body'],
               'nodes': [{'extensions': {'OMI_physics_body': {
                   'trigger': {'shape': 0, 'collisionFilter': 2}}}}]}
        physics = omi_gltf.load_document(doc)
        assert physics.node_bodies[0].trigger == model.Trigger(shape=0, collisionFilter=2)
        _top, nodes = omi_gltf.export_extensions(physics)
        assert nodes[0]['OMI_physics_body']['trigger'] == {'shape': 0, 'collisionFilter': 2}


class TestThreaded:
    def test_events_are_published_with_the_snapshot(self) -> None:
        world, material, _floor = world_with_floor(restitution=0.6)
        drop(world, material, model.Shape.sphere(0.2), height=0.4)
        world.contact_reporting = 'all'
        sim = ThreadedSimulation(world, sim_hz=120.0)
        with sim.with_world():
            for _ in range(30):
                world.step(STEP)
                sim._publish()
        snap, version, events = sim.latest_and_events()
        assert snap is not None and version == 30
        assert begins(events)
        assert sim.drain_events() == []
        assert len(world.contact_log) == 0


class TestPreSolve:
    def pane_world(self, verdict):
        """A ball rolling at a flagged pane, and a filter answering ``verdict``."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=STEP,
                             sleep_enabled=False)
        pane_shape = world.add_shape(model.Shape.box((0.05, 2, 2)))
        pane = world.add_body(model.Motion(type=model.STATIC),
                              collider=model.Collider(shape=pane_shape),
                              position=(0, 0, 0), report_contacts=True)
        ball_shape = world.add_shape(model.Shape.sphere(0.2))
        ball = world.add_body(model.Motion(type=model.DYNAMIC, mass=2.0,
                                           linearVelocity=(5.0, 0, 0)),
                              collider=model.Collider(shape=ball_shape),
                              position=(-0.5, 0, 0))
        asked = []

        def decide(pair):
            asked.append(pair)
            return verdict(pair) if callable(verdict) else verdict

        world.set_contact_filter(decide)
        heard = []
        world.add_contact_listener(heard.append)
        return SimpleNamespace(world=world, pane=pane, ball=ball, asked=asked,
                               heard=heard)

    def test_an_ignored_pair_lets_the_ball_through(self, solver_path) -> None:
        scene = self.pane_world(Verdict.IGNORE_PAIR)
        for _ in range(60):
            scene.world.step(STEP)
        assert scene.world.linear_velocity[scene.ball][0] == pytest.approx(5.0)
        assert scene.world.position[scene.ball][0] > 1.0
        assert len(scene.asked) == 1
        assert [(e.phase, e.solved) for e in scene.heard] == [
            ('begin', False), ('end', False)]

    def test_the_preview_says_how_hard_the_blow_will_be(self, solver_path) -> None:
        scene = self.pane_world(Verdict.IGNORE_PAIR)
        world = scene.world
        for _ in range(10):
            world.step(STEP)
        pair = scene.asked[0]
        assert (pair.a, pair.b) == (world.ref(scene.pane), world.ref(scene.ball))
        assert pair.approach == pytest.approx(5.0)
        assert pair.mass == pytest.approx(2.0)
        assert pair.impulse == pytest.approx(10.0)
        # From the pane, index 0, towards the ball arriving from -x.
        assert np.allclose(pair.normal, (-1, 0, 0))

    def test_solving_is_the_default(self, solver_path) -> None:
        scene = self.pane_world(None)
        for _ in range(30):
            scene.world.step(STEP)
        assert scene.world.linear_velocity[scene.ball][0] < 0.5
        assert scene.heard[0].solved

    def test_ignoring_one_step_asks_again(self, solver_path) -> None:
        answers = iter([Verdict.IGNORE_STEP, Verdict.SOLVE])
        scene = self.pane_world(lambda pair: next(answers))
        for _ in range(30):
            scene.world.step(STEP)
        assert len(scene.asked) == 2
        assert scene.heard[0].phase == 'begin' and scene.heard[0].solved
        assert scene.heard[0].step == scene.asked[1].step

    def test_it_is_asked_only_about_pairs_that_are_reported(self, solver_path) -> None:
        world, material, _floor = world_with_floor()
        drop(world, material, model.Shape.sphere(0.2), height=0.3)
        asked = []
        world.contact_reporting = 'flagged'
        world.set_contact_filter(asked.append)
        for _ in range(30):
            world.step(STEP)
        assert asked == []


def test_the_module_says_what_it_exports() -> None:
    assert set(contactevents.__all__) >= {
        'BodyRef', 'ContactEvent', 'TriggerEvent', 'ContactLog', 'ContactTracker',
        'PairPreview', 'Verdict'}
