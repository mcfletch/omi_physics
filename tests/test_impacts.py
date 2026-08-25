"""How hard a pair met: :meth:`PhysicsWorld.impact_on` (no GL).

The number a game decides damage, a crash or an impact sound from. It has to be
taken at the moment the two touch, because resolving the contact is exactly
cancelling the velocity that measures it -- so what is tested here is mostly
that the reading survives the solver, and that it is about the *pair* rather
than about how fast one of them happened to be going.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.collide import Contact
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def world_with_ground(restitution=0.0):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    material = world.add_material(model.Material(restitution=restitution))
    ground = world.add_shape(model.Shape.box((80, 1, 80)))
    floor = world.add_body(
        model.Motion(type=model.STATIC),
        collider=model.Collider(shape=ground, physicsMaterial=material),
        position=(0, -0.5, 0))
    return world, material, floor


def box_body(world, material, position, velocity=(0, 0, 0),
             kind=model.DYNAMIC, size=(2.0, 1.0, 4.0)):
    shape = world.add_shape(model.Shape.box(size))
    body = world.add_body(
        model.Motion(type=kind, mass=1200.0),
        collider=model.Collider(shape=shape, physicsMaterial=material),
        position=position)
    world.linear_velocity[body] = np.asarray(velocity, dtype='d')
    return body


def run_until_struck(world, body, steps=600, **named):
    """Step until something hits ``body``; the impact, or None if nothing did."""
    for _ in range(steps):
        world.step(DT)
        struck = world.impact_on(body, **named)
        if struck is not None:
            return struck
    return None


class TestHowHardSomethingWasHit:
    def test_a_body_run_into_a_wall_reports_the_speed_it_arrived_at(self) -> None:
        world, material, _floor = world_with_ground()
        wall_shape = world.add_shape(model.Shape.box((1.0, 4.0, 20.0)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=wall_shape,
                                               physicsMaterial=material),
                       position=(20.0, 2.0, 0.0))
        runner = box_body(world, material, (0.0, 2.0, 0.0), (30.0, 0.0, 0.0))
        world.gravity_factor[runner] = 0.0
        struck = run_until_struck(world, runner)
        assert struck is not None, 'nothing was hit'
        _other, closing = struck
        assert closing == pytest.approx(30.0, abs=0.5)

    def test_it_says_what_was_hit(self) -> None:
        world, material, _floor = world_with_ground()
        parked = box_body(world, material, (16.0, 0.5, 0.0), kind=model.KINEMATIC)
        runner = box_body(world, material, (0.0, 0.5, 0.0), (25.0, 0.0, 0.0))
        struck = run_until_struck(world, runner, skip_static=True)
        assert struck is not None and struck[0] == parked

    def test_catching_something_up_reports_the_difference_not_the_speed(self) -> None:
        """Two bodies going the same way meet at the difference between them.

        A car doing thirty that touches one doing twenty-eight has been hit at
        two, and a rule that read one speed rather than the pair would end
        every run at the first vehicle anybody drew alongside.
        """
        world, material, _floor = world_with_ground()
        ahead = box_body(world, material, (14.0, 0.5, 0.0), (28.0, 0.0, 0.0),
                         kind=model.KINEMATIC)
        runner = box_body(world, material, (0.0, 0.5, 0.0), (30.0, 0.0, 0.0))
        world.gravity_factor[runner] = 0.0
        for _ in range(1200):
            world.step(DT)
            world.linear_velocity[runner][0] = 30.0
            world.linear_velocity[ahead][0] = 28.0
            struck = world.impact_on(runner, skip_static=True)
            if struck is not None:
                assert struck[1] == pytest.approx(2.0, abs=0.6)
                return
        pytest.fail('the two never met')

    def test_a_resting_body_only_ever_reads_the_step_gravity_gave_it(self) -> None:
        """A body held against something is closing on it on every step: over
        one step gravity hands it ``g dt`` for the solver to take away again.
        That is the floor a caller telling a landing from a rest sets
        ``above`` past."""
        world, material, _floor = world_with_ground()
        resting = box_body(world, material, (0.0, 0.5, 0.0))
        for _ in range(400):
            world.step(DT)
        settled = world.impact_on(resting)
        assert settled is not None
        assert settled[1] == pytest.approx(9.81 * DT, abs=0.01)
        assert world.impact_on(resting, above=9.81 * DT * 2.0) is None

    def test_nothing_touching_reports_nothing(self) -> None:
        world, material, _floor = world_with_ground()
        falling = box_body(world, material, (0.0, 20.0, 0.0))
        world.step(DT)
        assert world.impact_on(falling) is None

    def test_the_ground_can_be_left_out_of_the_answer(self) -> None:
        """What a caller asking "what did I hit" means, landing being neither."""
        world, material, _floor = world_with_ground()
        dropped = box_body(world, material, (0.0, 6.0, 0.0))
        landed = run_until_struck(world, dropped)
        assert landed is not None and landed[1] > 5.0
        assert run_until_struck(world, dropped, steps=1, skip_static=True) is None

    def test_the_reading_is_taken_before_the_solver_cancels_it(self) -> None:
        """The point of the whole thing: a square-on hit is resolved by
        removing precisely the velocity that measures how hard it was, so a
        reading taken afterwards is a reading of nothing."""
        world, material, _floor = world_with_ground()
        wall_shape = world.add_shape(model.Shape.box((1.0, 4.0, 20.0)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=wall_shape,
                                               physicsMaterial=material),
                       position=(20.0, 2.0, 0.0))
        runner = box_body(world, material, (0.0, 2.0, 0.0), (40.0, 0.0, 0.0))
        world.gravity_factor[runner] = 0.0
        struck = run_until_struck(world, runner)
        assert struck is not None
        assert struck[1] > 30.0
        # and afterwards there is nothing left to read
        assert abs(float(world.linear_velocity[runner][0])) < struck[1]


class TestWhichBlowIsAnsweredWith:
    """Only the heaviest is, so a caller with one question in mind has to be
    able to say which bodies it is about: a car that clipped a parapet and a
    rival in the same step is otherwise told about the parapet and drives on.

    Contacts made by hand rather than driven into being, because what is under
    test is which of a step's contacts is picked out and not what produces
    them.
    """

    @staticmethod
    def _struck(world, mine, *blows):
        """``mine`` hit by each ``(other, closing)``, in one step's contacts."""
        world.contacts = [
            Contact(mine, other, np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.01,
                    approach=float(closing))
            for other, closing in blows]
        return world

    def test_the_hardest_one_by_default(self) -> None:
        world, material, _floor = world_with_ground()
        near = box_body(world, material, (4.0, 0.5, 0.0), kind=model.KINEMATIC)
        far = box_body(world, material, (8.0, 0.5, 0.0), kind=model.KINEMATIC)
        mine = box_body(world, material, (0.0, 0.5, 0.0))
        self._struck(world, mine, (near, 4.0), (far, 25.0))
        assert world.impact_on(mine) == (far, 25.0)

    def test_the_hardest_among_the_ones_asked_about(self) -> None:
        world, material, _floor = world_with_ground()
        near = box_body(world, material, (4.0, 0.5, 0.0), kind=model.KINEMATIC)
        far = box_body(world, material, (8.0, 0.5, 0.0), kind=model.KINEMATIC)
        mine = box_body(world, material, (0.0, 0.5, 0.0))
        self._struck(world, mine, (near, 4.0), (far, 25.0))
        assert world.impact_on(mine, among={near}) == (near, 4.0)

    def test_and_nothing_when_none_of_them_was_touched(self) -> None:
        world, material, _floor = world_with_ground()
        near = box_body(world, material, (4.0, 0.5, 0.0), kind=model.KINEMATIC)
        elsewhere = box_body(world, material, (40.0, 0.5, 0.0),
                             kind=model.KINEMATIC)
        mine = box_body(world, material, (0.0, 0.5, 0.0))
        self._struck(world, mine, (near, 25.0))
        assert world.impact_on(mine, among={elsewhere}) is None

    def test_a_blow_landed_on_something_else_is_not_mine(self) -> None:
        world, material, _floor = world_with_ground()
        near = box_body(world, material, (4.0, 0.5, 0.0), kind=model.KINEMATIC)
        other = box_body(world, material, (8.0, 0.5, 0.0))
        mine = box_body(world, material, (0.0, 0.5, 0.0))
        self._struck(world, other, (near, 25.0))
        assert world.impact_on(mine) is None
        assert world.impact_on(other) == (near, 25.0)
