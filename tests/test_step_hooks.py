"""Callbacks the world makes around every fixed step.

``PhysicsWorld.add_step_hook`` registers a ``before`` and an ``after`` function.
Each receives a :class:`~omi_physics.world.Step` naming the step: its ``dt``,
its ``index`` (the ``step`` every contact event of that step carries) and the
simulation ``time`` at its start. They run on whichever thread steps the world.
"""
import threading

import pytest

from omi_physics import model
from omi_physics.threaded import ThreadedSimulation
from omi_physics.world import PhysicsWorld, Step

STEP = 1.0 / 120.0


def falling_world() -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), fixed_dt=STEP,
                         sleep_enabled=False)
    floor_shape = world.add_shape(model.Shape.box((20, 1, 20)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=floor_shape),
                   position=(0, -0.5, 0))
    ball_shape = world.add_shape(model.Shape.sphere(0.25))
    world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                   collider=model.Collider(shape=ball_shape),
                   position=(0, 0.4, 0))
    return world


class TestStepHooks:
    def test_each_step_calls_before_then_after_with_its_record(self) -> None:
        world = falling_world()
        seen: list[tuple[str, Step, int]] = []
        world.add_step_hook(before=lambda s: seen.append(('before', s, world.step_count)),
                            after=lambda s: seen.append(('after', s, world.step_count)))
        world.step(STEP)
        world.step(STEP)
        assert [(kind, rec.index, count) for kind, rec, count in seen] == [
            ('before', 0, 0), ('after', 0, 1), ('before', 1, 1), ('after', 1, 2)]
        assert seen[0][1] == Step(dt=STEP, index=0, time=0.0)
        assert seen[2][1].time == pytest.approx(STEP)

    def test_advance_calls_the_hooks_once_per_fixed_step(self) -> None:
        world = falling_world()
        indexes: list[int] = []
        world.add_step_hook(after=lambda s: indexes.append(s.index))
        world.advance(3.5 * STEP)
        assert indexes == [0, 1, 2]

    def test_the_remover_ends_the_subscription_and_is_repeatable(self) -> None:
        world = falling_world()
        calls: list[int] = []
        remove = world.add_step_hook(before=lambda s: calls.append(s.index))
        world.step(STEP)
        remove()
        remove()
        world.step(STEP)
        assert calls == [0]

    def test_hooks_run_in_registration_order(self) -> None:
        world = falling_world()
        order: list[str] = []
        world.add_step_hook(before=lambda s: order.append('first'))
        world.add_step_hook(before=lambda s: order.append('second'))
        world.step(STEP)
        assert order == ['first', 'second']

    def test_a_hook_that_raises_does_not_stop_the_step(self) -> None:
        world = falling_world()
        later: list[int] = []

        def broken(step: Step) -> None:
            raise RuntimeError('hook failed')

        world.add_step_hook(before=broken)
        world.add_step_hook(after=lambda s: later.append(s.index))
        with pytest.raises(RuntimeError, match='hook failed'):
            world.step(STEP)
        assert world.step_count == 1
        assert later == [0]

    def test_a_hook_can_remove_itself(self) -> None:
        world = falling_world()
        calls: list[int] = []
        holder: list = []

        def once(step: Step) -> None:
            calls.append(step.index)
            holder[0]()

        holder.append(world.add_step_hook(before=once))
        world.step(STEP)
        world.step(STEP)
        assert calls == [0]

    def test_a_hook_can_act_on_the_world_it_is_called_for(self) -> None:
        """A force set in ``before`` is in the step that follows it."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=STEP,
                             sleep_enabled=False)
        shape = world.add_shape(model.Shape.sphere(0.25))
        ball = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                              collider=model.Collider(shape=shape))
        world.add_step_hook(before=lambda s: world.apply_impulse(ball, (1.0, 0, 0)))
        world.step(STEP)
        assert world.linear_velocity[ball][0] == pytest.approx(1.0)


class TestEventsCarryTheStep:
    def test_a_contact_event_has_the_index_and_time_of_the_step_that_made_it(self) -> None:
        world = falling_world()
        world.contact_reporting = 'all'
        records: dict[int, Step] = {}
        world.add_step_hook(after=lambda s: records.__setitem__(s.index, s))
        for _ in range(60):
            world.step(STEP)
        events = world.contact_log.drain()
        assert events
        for event in events:
            assert event.step in records
            assert event.time == pytest.approx(records[event.step].time)

    def test_a_threaded_run_publishes_events_in_the_order_of_their_steps(self) -> None:
        world = falling_world()
        world.contact_reporting = 'all'
        ran_on: set[int] = set()
        indexes: list[int] = []

        def after(step: Step) -> None:
            ran_on.add(threading.get_ident())
            indexes.append(step.index)

        world.add_step_hook(after=after)
        sim = ThreadedSimulation(world, sim_hz=240.0)
        sim.start()
        try:
            assert sim.wait_steps(60, timeout=10)
        finally:
            sim.stop()
        events = sim.drain_events()
        assert events
        steps = [event.step for event in events]
        assert steps == sorted(steps)
        assert set(steps) <= set(indexes)
        assert len(ran_on) == 1
        assert threading.get_ident() not in ran_on
