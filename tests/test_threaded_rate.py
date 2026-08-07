"""Whether the simulation thread is keeping up, which nothing else can see.

``ThreadedSimulation`` asks for a fixed tick rate and silently settles for
whatever it gets: a tick that overruns its budget resets the schedule rather
than trying to catch up, which is right -- catching up spirals -- and invisible.
A simulation being starved and a simulation running to time look identical from
the render thread, and identical on screen except that everything moves slowly.

These pin the two counters that tell them apart: the rate actually achieved,
and how many ticks were abandoned to stay out of the spiral.
"""
import time

import pytest

from omi_physics import model
from omi_physics.threaded import ThreadedSimulation
from omi_physics.world import PhysicsWorld


def _world():
    return PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))


class TestTheRecentRate:
    """Arithmetic over tick timestamps, checked against timestamps by hand."""

    def test_ticks_a_hundredth_apart_are_a_hundred_a_second(self):
        sim = ThreadedSimulation(_world(), sim_hz=100.0)
        for index in range(11):
            sim._record_tick(index * 0.01)
        assert sim.rate() == pytest.approx(100.0)

    def test_a_starved_thread_reports_the_rate_it_achieved(self):
        """Asked for 120Hz, given one tick a second: the report is 1, not 120."""
        sim = ThreadedSimulation(_world(), sim_hz=120.0)
        for index in range(5):
            sim._record_tick(float(index))
        assert sim.rate() == pytest.approx(1.0)

    def test_the_window_is_bounded_and_reports_only_the_recent_past(self):
        sim = ThreadedSimulation(_world(), sim_hz=100.0)
        for index in range(500):            # long ago, and slow
            sim._record_tick(index * 1.0)
        base = 500.0
        for index in range(1, sim.RATE_WINDOW + 1):   # lately, and fast
            sim._record_tick(base + index * 0.01)
        assert len(sim._tick_times) <= sim.RATE_WINDOW
        assert sim.rate() == pytest.approx(100.0, rel=0.02)

    def test_nothing_ticked_yet_is_no_rate_rather_than_a_division(self):
        sim = ThreadedSimulation(_world(), sim_hz=100.0)
        assert sim.rate() == 0.0
        sim._record_tick(1.0)
        assert sim.rate() == 0.0            # one stamp is not an interval

    def test_two_ticks_at_the_same_instant_do_not_divide_by_zero(self):
        sim = ThreadedSimulation(_world(), sim_hz=100.0)
        sim._record_tick(1.0)
        sim._record_tick(1.0)
        assert sim.rate() == 0.0


class TestDroppedTicks:
    def test_a_fresh_simulation_has_dropped_nothing(self):
        assert ThreadedSimulation(_world(), sim_hz=100.0).dropped == 0

    def test_a_tick_that_overruns_its_budget_is_counted(self):
        """A world too slow for the asked-for rate: every tick is late."""
        class SlowWorld(PhysicsWorld):
            def step(self, dt):
                time.sleep(0.01)
                return super().step(dt)

        sim = ThreadedSimulation(
            SlowWorld(gravity=model.Gravity(gravity=0.0)), sim_hz=400.0)
        sim.start()
        try:
            deadline = time.time() + 2.0
            while sim.steps < 5 and time.time() < deadline:
                time.sleep(0.005)
        finally:
            sim.stop()
        assert sim.steps >= 5
        # 400Hz asks for a tick every 2.5ms and each takes 10ms, so every tick
        # bar the first overruns.
        assert sim.dropped >= sim.steps - 1

    def test_a_simulation_that_keeps_up_drops_nothing(self):
        sim = ThreadedSimulation(_world(), sim_hz=60.0)
        sim.start()
        try:
            deadline = time.time() + 2.0
            while sim.steps < 5 and time.time() < deadline:
                time.sleep(0.005)
        finally:
            sim.stop()
        assert sim.steps >= 5
        assert sim.dropped == 0


class TestOnARunningThread:
    def test_the_reported_rate_tracks_the_rate_asked_for(self):
        sim = ThreadedSimulation(_world(), sim_hz=100.0)
        sim.start()
        try:
            deadline = time.time() + 3.0
            while sim.steps < 30 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            sim.stop()
        assert sim.steps >= 30
        # Generous: a shared CI box schedules a daemon thread when it feels
        # like it. The test is that the number is real, not that it is exact.
        assert sim.rate() == pytest.approx(100.0, rel=0.35)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
