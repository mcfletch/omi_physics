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
from omi_physics.threaded import ThreadedSimulation, pace
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


class TestPacing:
    """The rule the loop schedules by, given clock readings by hand.

    Whether a real thread is handed its turn inside 16ms is the operating
    system's decision and not this library's, so the rule is checked here
    against times chosen for it rather than against times a machine produced.
    """

    def test_a_tick_inside_its_budget_waits_out_the_remainder(self):
        delay, due, dropped = pace(next_t=1.0, now=1.004, dt=0.01)
        assert delay == pytest.approx(0.006)
        assert due == pytest.approx(1.01)
        assert not dropped

    def test_a_tick_that_took_exactly_its_budget_drops(self):
        """No time left is not time to spare: nothing to wait out, and the
        next tick starts already at its deadline."""
        delay, due, dropped = pace(next_t=1.0, now=1.01, dt=0.01)
        assert delay == 0.0
        assert due == pytest.approx(1.01)
        assert dropped

    def test_an_overrun_is_dropped_and_the_deadline_moves_to_now(self):
        """The debt is forgiven rather than carried: a machine that is behind
        cannot pay back missed ticks, and trying is the spiral."""
        delay, due, dropped = pace(next_t=1.0, now=1.5, dt=0.01)
        assert delay == 0.0
        assert due == 1.5
        assert dropped

    def test_a_run_of_healthy_ticks_holds_the_asked_for_cadence(self):
        """Deadlines advance by exactly dt, so small early or late arrivals
        do not accumulate into drift."""
        due = 0.0
        for index in range(100):
            # Each tick finishes a jittery but comfortable way into its budget.
            now = due + 0.01 * (0.3 + 0.4 * ((index * 7) % 5) / 4.0)
            _delay, due, dropped = pace(due, now, 0.01)
            assert not dropped
        assert due == pytest.approx(1.0)

    def test_falling_behind_does_not_accumulate_a_debt_of_ticks(self):
        """After a long stall the next tick is due one dt away, not a hundred
        ticks in the past."""
        _, due, dropped = pace(next_t=1.0, now=2.0, dt=0.01)
        assert dropped
        delay, due, dropped = pace(due, now=2.001, dt=0.01)
        assert not dropped
        assert delay == pytest.approx(0.009)


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

    def test_a_running_thread_counts_no_more_drops_than_it_took_ticks(self):
        """The loop counts a drop only where :func:`pace` reports one.

        How many of an empty world's 60Hz ticks a loaded machine actually
        hands over on time is the operating system's business -- what is this
        library's is that the count means something. ``TestPacing`` pins when a
        drop is counted; this pins that the loop is the thing doing the
        counting.
        """
        sim = ThreadedSimulation(_world(), sim_hz=60.0)
        sim.start()
        try:
            deadline = time.time() + 2.0
            while sim.steps < 5 and time.time() < deadline:
                time.sleep(0.005)
        finally:
            sim.stop()
        assert sim.steps >= 5
        assert 0 <= sim.dropped <= sim.steps


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
