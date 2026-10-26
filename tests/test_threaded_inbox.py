"""The simulation thread's command inbox and its idle block.

Commands are functions of the world. ``post`` queues one from any thread and
``call`` queues one and answers a future. Both run on the simulation thread at
the start of a tick, before that tick's step, in the order they were queued.
With nothing awake and nothing queued the thread blocks on an event instead of
ticking.
"""
import threading
import time
from concurrent.futures import Future

import pytest

from omi_physics import model
from omi_physics.threaded import ThreadedSimulation
from omi_physics.world import PhysicsWorld

HZ = 240.0


def resting_world() -> tuple[PhysicsWorld, int]:
    """No gravity, one ball at rest that falls asleep after 10 ms."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=1.0 / HZ)
    world.sleep_time = 0.01
    shape = world.add_shape(model.Shape.sphere(0.25))
    ball = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape))
    return world, ball


@pytest.fixture
def moving() -> ThreadedSimulation:
    """A simulation whose ball keeps moving, so it never goes idle."""
    world, ball = resting_world()
    world.linear_velocity[ball] = (1.0, 0.0, 0.0)
    simulation = ThreadedSimulation(world, sim_hz=HZ)
    yield simulation
    simulation.stop()


@pytest.fixture
def sim() -> ThreadedSimulation:
    world, _ball = resting_world()
    simulation = ThreadedSimulation(world, sim_hz=HZ)
    yield simulation
    simulation.stop()


def wait_until_idle(sim: ThreadedSimulation, window: float = 0.05) -> None:
    """Return once no tick has run for ``window`` seconds."""
    deadline = time.monotonic() + 10
    last = -1
    while time.monotonic() < deadline:
        steps = sim.steps
        if steps == last:
            return
        last = steps
        time.sleep(window)
    raise AssertionError('the simulation never went idle')


class TestInbox:
    def test_call_runs_on_the_simulation_thread_and_returns_the_result(self, sim) -> None:
        sim.start()
        future = sim.call(lambda world: (threading.get_ident(), world is sim.world))
        ident, is_world = future.result(timeout=5)
        assert is_world
        assert ident != threading.get_ident()

    def test_call_returns_a_future(self, sim) -> None:
        sim.start()
        assert isinstance(sim.call(lambda world: 1), Future)

    def test_call_carries_an_exception_to_the_future(self, sim) -> None:
        sim.start()

        def fail(world: PhysicsWorld) -> None:
            raise KeyError('missing')

        with pytest.raises(KeyError):
            sim.call(fail).result(timeout=5)
        assert sim.call(lambda world: 'still running').result(timeout=5) == 'still running'

    def test_a_posted_command_that_raises_does_not_end_the_thread(self, sim, caplog) -> None:
        sim.start()

        def fail(world: PhysicsWorld) -> None:
            raise ValueError('posted failure')

        sim.post(fail)
        assert sim.call(lambda world: 'alive').result(timeout=5) == 'alive'
        assert 'posted failure' in caplog.text

    def test_commands_run_in_post_order(self, sim) -> None:
        order: list[int] = []
        for number in range(50):
            sim.post(lambda world, number=number: order.append(number))
        sim.start()
        sim.call(lambda world: None).result(timeout=5)
        assert order == list(range(50))

    def test_posts_from_many_threads_all_run(self, sim) -> None:
        seen: list[int] = []
        sim.start()

        def poster(base: int) -> None:
            for number in range(100):
                sim.post(lambda world, value=base + number: seen.append(value))

        threads = [threading.Thread(target=poster, args=(base * 100,)) for base in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        sim.call(lambda world: None).result(timeout=5)
        assert sorted(seen) == list(range(800))

    def test_a_command_runs_before_the_step_of_its_tick(self, moving) -> None:
        log: list[tuple[str, int]] = []
        moving.world.add_step_hook(before=lambda step: log.append(('step', step.index)))
        moving.start()
        assert moving.wait_steps(3, timeout=5)
        moving.call(lambda world: log.append(('command', world.step_count))).result(timeout=5)
        assert moving.wait_steps(moving.steps + 2, timeout=5)
        position = next(at for at, entry in enumerate(log) if entry[0] == 'command')
        assert log[position + 1] == ('step', log[position][1])

    def test_a_command_posted_by_a_command_runs_on_the_next_tick(self, moving) -> None:
        marks: list[int] = []

        def outer(world: PhysicsWorld) -> None:
            marks.append(world.step_count)
            moving.post(lambda inner: marks.append(inner.step_count))

        moving.start()
        moving.call(outer).result(timeout=5)
        assert moving.wait_steps(moving.steps + 3, timeout=5)
        assert len(marks) == 2
        assert marks[1] > marks[0]

    def test_commands_posted_before_start_wait_for_it(self, sim) -> None:
        future = sim.call(lambda world: 'ran')
        assert not future.done()
        sim.start()
        assert future.result(timeout=5) == 'ran'

    def test_call_from_the_simulation_thread_runs_at_once(self, sim) -> None:
        sim.start()

        def nested(world: PhysicsWorld) -> str:
            return sim.call(lambda inner: 'inline').result(timeout=1)

        assert sim.call(nested).result(timeout=5) == 'inline'

    def test_leaving_with_world_wakes_an_idle_simulation(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        with sim.with_world() as world:
            shape = world.add_shape(model.Shape.sphere(0.1))
            world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0,
                                        linearVelocity=(1, 0, 0)),
                           collider=model.Collider(shape=shape))
        before = sim.steps
        assert sim.wait_steps(before + 3, timeout=5)


class TestIdleBlock:
    def test_ticks_stop_while_nothing_is_awake(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        before = sim.steps
        time.sleep(0.3)
        assert sim.steps == before

    def test_a_posted_velocity_resumes_ticking(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        before = sim.steps
        sim.post(lambda world: world.apply_impulse(0, (1.0, 0.0, 0.0)))
        assert sim.wait_steps(before + 10, timeout=5)
        assert sim.world.position[0][0] > 0.0

    def test_wake_resumes_ticking_after_a_change_made_outside_the_inbox(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        before = sim.steps
        sim.world.linear_velocity[0] = (1.0, 0.0, 0.0)
        sim.world.wake(0)
        sim.wake()
        assert sim.wait_steps(before + 5, timeout=5)

    def test_a_moving_kinematic_body_keeps_the_thread_ticking(self) -> None:
        world, _ball = resting_world()
        shape = world.add_shape(model.Shape.box((1, 1, 1)))
        world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(0.0, 1.0, 0.0)),
                       collider=model.Collider(shape=shape), position=(5, 0, 0))
        sim = ThreadedSimulation(world, sim_hz=HZ)
        sim.start()
        try:
            time.sleep(0.2)
            before = sim.steps
            time.sleep(0.2)
            assert sim.steps > before + 20
        finally:
            sim.stop()

    def test_a_stationary_kinematic_body_lets_it_idle(self) -> None:
        world, _ball = resting_world()
        shape = world.add_shape(model.Shape.box((1, 1, 1)))
        world.add_body(model.Motion(type=model.KINEMATIC),
                       collider=model.Collider(shape=shape), position=(5, 0, 0))
        sim = ThreadedSimulation(world, sim_hz=HZ)
        sim.start()
        try:
            wait_until_idle(sim)
            before = sim.steps
            time.sleep(0.2)
            assert sim.steps == before
        finally:
            sim.stop()

    def test_stop_interrupts_an_idle_wait(self, sim) -> None:
        before = set(threading.enumerate())
        sim.start()
        wait_until_idle(sim)
        (thread,) = set(threading.enumerate()) - before
        began = time.monotonic()
        sim.stop()
        assert time.monotonic() - began < 1.0
        assert not thread.is_alive()

    def test_an_idle_simulation_does_not_count_the_wait_as_dropped_ticks(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        time.sleep(0.1)
        before = sim.steps
        sim.post(lambda world: world.apply_impulse(0, (1.0, 0.0, 0.0)))
        assert sim.wait_steps(before + 20, timeout=5)
        assert sim.dropped == 0

    def test_the_rate_window_forgets_the_idle_gap(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        time.sleep(0.3)
        before = sim.steps
        sim.post(lambda world: world.apply_impulse(0, (1.0, 0.0, 0.0)))
        assert sim.wait_steps(before + 40, timeout=5)
        assert sim.rate() > HZ * 0.5

    def test_idle_blocking_can_be_switched_off(self) -> None:
        world, _ball = resting_world()
        sim = ThreadedSimulation(world, sim_hz=HZ, block_when_idle=False)
        sim.start()
        try:
            time.sleep(0.2)
            before = sim.steps
            time.sleep(0.2)
            assert sim.steps > before + 20
        finally:
            sim.stop()

    def test_a_command_that_changes_a_static_pose_republishes_while_idle(self, sim) -> None:
        sim.start()
        wait_until_idle(sim)
        version = sim.latest()[1]
        sim.call(lambda world: world.place_body(0, position=(3.0, 0.0, 0.0))).result(timeout=5)
        snapshot, newer = sim.latest()
        assert newer > version
        assert snapshot is not None and snapshot[0][0][0] == pytest.approx(3.0)


class TestStart:
    def test_two_concurrent_starts_make_one_thread(self) -> None:
        class SlowWorld(PhysicsWorld):
            """Reading the body count takes a while, so a second start can interleave."""

            @property
            def body_count(self) -> int:
                time.sleep(0.05)
                return super().body_count

        world = SlowWorld(gravity=model.Gravity(gravity=9.81))
        shape = world.add_shape(model.Shape.sphere(0.25))
        world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                       collider=model.Collider(shape=shape), position=(0, 100, 0))
        stepping: set[int] = set()
        world.add_step_hook(before=lambda step: stepping.add(threading.get_ident()))
        sim = ThreadedSimulation(world, sim_hz=HZ)
        barrier = threading.Barrier(4)

        def starter() -> None:
            barrier.wait()
            sim.start()

        threads = [threading.Thread(target=starter) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        try:
            assert sim.wait_steps(20, timeout=10)
        finally:
            sim.stop()
        assert len(stepping) == 1

    def test_the_simulation_restarts_after_a_stop(self, sim) -> None:
        sim.start()
        sim.stop()
        sim.start()
        assert sim.call(lambda world: 'again').result(timeout=5) == 'again'


class TestStopAndStart:
    def test_a_stop_from_a_command_lets_a_later_start_run_again(self, sim) -> None:
        sim.start()
        sim.call(lambda world: sim.stop()).result(timeout=5)
        deadline = time.monotonic() + 5
        while sim._thread is not None and time.monotonic() < deadline:  # noqa: SLF001 the thread's exit is what is tested
            time.sleep(0.01)
        sim.start()
        assert sim.call(lambda world: 'again').result(timeout=5) == 'again'

    def test_a_stop_that_times_out_leaves_one_loop_stepping(self, moving) -> None:
        moving.STOP_TIMEOUT = 0.05
        stepping: set[int] = set()
        moving.world.add_step_hook(before=lambda step: stepping.add(threading.get_ident()))
        held = threading.Event()
        release = threading.Event()

        def block(world: PhysicsWorld) -> None:
            held.set()
            release.wait(5)

        moving.start()
        moving.post(block)
        assert held.wait(5)
        moving.stop()
        # start() publishes under the world lock, which the blocked command holds.
        threading.Timer(0.1, release.set).start()
        moving.start()
        time.sleep(0.1)
        stepping.clear()
        assert moving.wait_steps(moving.steps + 20, timeout=10)
        assert len(stepping) == 1


class TestCallDuringAStep:
    def test_a_call_from_a_step_hook_waits_for_the_next_tick(self, moving) -> None:
        seen: list[bool] = []
        futures: list[Future] = []

        def hook(step) -> None:
            if not futures:
                future = moving.call(lambda world: world.step_count)
                futures.append(future)
                seen.append(future.done())

        moving.world.add_step_hook(after=hook)
        moving.start()
        assert moving.wait_steps(3, timeout=10)
        assert seen == [False]
        assert futures[0].result(timeout=5) >= 1
