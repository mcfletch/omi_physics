"""A character and the dynamic bodies around it act on each other.

The controller moves its capsule against a snapshot of the world and does not
push anything itself. A kinematic stand-in in the world, placed each step the
way the physics service places it, carries the character's velocity into the
solver. ``Rig.advance`` is that placement. The stand-in is wider than the
capsule by ``SKIN``, so it overlaps a body the capsule touches: the solver acts
on overlaps, and a stand-in placed exactly at the capsule's surface would never
meet the body it is meant to push.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0
GRAVITY = 9.81
SKIN = CharacterController.PUSH_REACH


class Rig:
    """A floor, a character with a kinematic stand-in, and whatever the test adds."""

    def __init__(self, position: tuple[float, float, float] = (0.0, 0.9, 0.0),
                 **caps: float | None) -> None:
        self.world = PhysicsWorld(gravity=model.Gravity(gravity=GRAVITY, direction=(0, -1, 0)),
                                  fixed_dt=DT)
        floor = self.world.add_shape(model.Shape.box((200, 1, 200)))
        self.world.add_body(model.Motion(type=model.STATIC),
                            collider=model.Collider(shape=floor), position=(0, -0.5, 0))
        self.caps = CharacterCapabilities(**caps)
        capsule = self.world.add_shape(model.Shape.capsule(
            height=self.caps.standHeight - 2 * self.caps.radius,
            radius=self.caps.radius + SKIN))
        self.stand_in = self.world.add_body(
            model.Motion(type=model.KINEMATIC), collider=model.Collider(shape=capsule),
            position=position)
        self.character = CharacterController(self.world, self.caps, position=position,
                                             gravity=GRAVITY, ignore=[self.stand_in])
        self.pushes: list[int] = []
        self.world.set_pair_rule(self.rule)

    def rule(self, i: int, j: int) -> bool | None:
        """The stand-in does not touch a body too heavy for the character to push."""
        if self.stand_in not in (i, j):
            return None
        other = j if i == self.stand_in else i
        return False if self.world.mass[other] > self.caps.pushMass else None

    def crate(self, mass: float, position: tuple[float, float, float],
              size: float = 0.8, velocity: tuple[float, float, float] = (0, 0, 0),
              gravity_factor: float = 1.0) -> int:
        shape = self.world.add_shape(model.Shape.box((size, size, size)))
        return self.world.add_body(
            model.Motion(type=model.DYNAMIC, mass=mass, linearVelocity=velocity,
                         gravityFactor=gravity_factor),
            collider=model.Collider(shape=shape), position=position)

    def advance(self, steps: int = 1) -> None:
        for _ in range(steps):
            character = self.character
            character.update(DT)
            pushing = character.pushing
            if pushing is not None:
                self.pushes.append(pushing[0])
            velocity = pushing[1] if pushing is not None else character.velocity
            self.world.place_body(self.stand_in, position=character.position)
            self.world.linear_velocity[self.stand_in] = velocity
            if float(np.linalg.norm(velocity)) > 0.0:
                self.world.wake(self.stand_in)
            self.world.step(DT)

    def settle(self, steps: int = 30) -> None:
        self.advance(steps)
        assert self.character.grounded

    def x_of(self, body: int) -> float:
        return float(self.world.position[body][0])


class TestCapabilities:
    def test_defaults(self) -> None:
        caps = CharacterCapabilities()
        assert caps.mass == 80.0
        assert caps.pushMass == 200.0
        assert caps.pushSpeed is None

    def test_a_push_speed_of_none_is_the_walk_speed(self) -> None:
        rig = Rig(walkSpeed=2.0)
        rig.settle()
        crate = rig.crate(20.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'run')
        rig.advance(240)
        assert float(rig.world.linear_velocity[crate][0]) == pytest.approx(2.0, abs=0.3)


class TestPushing:
    def test_a_walker_pushes_a_light_crate_at_walking_speed(self) -> None:
        rig = Rig()
        rig.settle()
        crate = rig.crate(20.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(60)
        start = rig.x_of(crate)
        rig.advance(120)
        assert (rig.x_of(crate) - start) / 2.0 == pytest.approx(3.0, rel=0.15)
        assert rig.pushes
        assert set(rig.pushes) == {crate}

    def test_the_walker_stays_against_the_crate_it_pushes(self) -> None:
        rig = Rig()
        rig.settle()
        crate = rig.crate(20.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(240)
        gap = (rig.x_of(crate) - 0.4) - (float(rig.character.position[0]) + 0.3)
        assert -0.01 < gap < 0.1

    def test_a_walker_does_not_move_a_heavy_crate(self) -> None:
        rig = Rig()
        rig.settle()
        crate = rig.crate(2000.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(240)
        assert rig.x_of(crate) == pytest.approx(1.5, abs=0.02)
        assert rig.character.pushing is None
        assert not rig.pushes
        assert float(rig.character.position[0]) < 1.5 - 0.4 - 0.3 + 0.02

    def test_push_mass_decides_which_crates_are_light(self) -> None:
        rig = Rig(pushMass=1000.0)
        rig.settle()
        crate = rig.crate(500.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(240)
        assert rig.x_of(crate) > 2.0
        assert float(rig.character.position[0]) < rig.x_of(crate)

    def test_push_speed_caps_the_speed_it_is_pushed_at(self) -> None:
        rig = Rig(pushSpeed=1.0)
        rig.settle()
        crate = rig.crate(20.0, (1.5, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'run')
        rig.advance(120)
        assert float(rig.world.linear_velocity[crate][0]) == pytest.approx(1.0, abs=0.25)

    def test_the_wanted_velocity_is_what_the_character_exposes(self) -> None:
        rig = Rig()
        rig.settle()
        crate = rig.crate(20.0, (0.7, 0.4, 0.0))
        rig.character.set_move((1, 0, 0), 'walk')
        seen = []
        for _ in range(60):
            rig.character.update(DT)
            if rig.character.pushing is not None:
                seen.append(rig.character.pushing)
        assert seen
        body, velocity = seen[0]
        assert body == crate
        assert velocity == pytest.approx([3.0, 0.0, 0.0])

    def test_nothing_is_pushed_in_open_ground(self) -> None:
        rig = Rig()
        rig.settle()
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(60)
        assert rig.character.pushing is None


class TestBlockingAndFloors:
    def test_a_crate_that_cannot_be_pushed_blocks_the_step(self) -> None:
        rig = Rig()
        rig.settle()
        rig.crate(2000.0, (1.5, 0.5, 0.0), size=1.0)
        rig.character.set_move((1, 0, 0), 'walk')
        rig.advance(120)
        assert float(rig.character.position[0]) < 1.0 - 0.3 + 0.05

    def test_a_character_lands_and_stands_on_a_crate(self) -> None:
        rig = Rig(position=(0.0, 3.0, 0.0))
        crate = rig.crate(2000.0, (0.0, 0.5, 0.0), size=1.0)
        rig.advance(120)
        assert rig.character.grounded
        assert float(rig.character.base()[1]) == pytest.approx(1.0, abs=0.02)
        assert rig.character.standing_on == crate

    def test_standing_on_the_floor_is_standing_on_nothing(self) -> None:
        rig = Rig()
        rig.settle()
        assert rig.character.standing_on is None

    def test_a_walker_on_a_drifting_platform_is_carried(self) -> None:
        rig = Rig(position=(0.0, 3.4, 0.0))
        raft = rig.crate(500.0, (0.0, 1.5, 0.0), size=2.0, velocity=(1.0, 0.0, 0.0),
                         gravity_factor=0.0)
        rig.advance(60)
        assert rig.character.grounded
        assert rig.character.standing_on == raft
        offset = float(rig.character.position[0]) - rig.x_of(raft)
        rig.advance(180)
        assert rig.x_of(raft) > 3.0
        assert float(rig.character.position[0]) - rig.x_of(raft) == pytest.approx(
            offset, abs=0.05)
        assert rig.character.grounded
        assert rig.character.velocity[0] == pytest.approx(1.0, abs=0.1)

    def test_a_character_walking_off_the_back_of_a_platform_falls(self) -> None:
        rig = Rig(position=(0.0, 3.4, 0.0))
        raft = rig.crate(500.0, (0.0, 1.5, 0.0), size=2.0, velocity=(1.0, 0.0, 0.0),
                         gravity_factor=0.0)
        rig.advance(60)
        rig.character.set_move((-1, 0, 0), 'run')
        rig.advance(120)
        assert rig.character.standing_on is None
        assert float(rig.character.position[0]) < rig.x_of(raft) - 1.0


class TestKnockback:
    def test_it_changes_the_velocity_by_impulse_over_mass(self) -> None:
        character = CharacterController(PhysicsWorld(), CharacterCapabilities(),
                                        position=(0.0, 50.0, 0.0), gravity=GRAVITY)
        character.update(DT)
        character.update(DT)
        before = character.velocity.copy()
        character.knockback((160.0, 80.0, -240.0))
        character.update(DT)
        gained = character.velocity - before
        assert gained[0] == pytest.approx(2.0, abs=1e-9)
        assert gained[2] == pytest.approx(-3.0, abs=1e-9)
        assert gained[1] == pytest.approx(1.0 - GRAVITY * DT, abs=1e-6)

    def test_it_uses_the_mass_in_the_capabilities(self) -> None:
        character = CharacterController(PhysicsWorld(), CharacterCapabilities(mass=40.0),
                                        position=(0.0, 50.0, 0.0), gravity=GRAVITY)
        character.update(DT)
        before = character.velocity.copy()
        character.knockback((80.0, 0.0, 0.0))
        character.update(DT)
        assert (character.velocity - before)[0] == pytest.approx(2.0, abs=1e-9)

    def test_an_upward_blow_lifts_a_grounded_character(self) -> None:
        rig = Rig()
        rig.settle()
        rig.character.knockback((0.0, 400.0, 0.0))
        assert not rig.character.grounded
        rig.advance(10)
        assert float(rig.character.base()[1]) > 0.1

    def test_a_thrown_crate_pushes_a_walker_back_by_the_momentum_it_gave_up(self) -> None:
        rig = Rig(position=(0.0, 0.9, 0.0))
        rig.settle()
        crate = rig.crate(20.0, (3.0, 0.4, 0.0), velocity=(-8.0, 0.0, 0.0),
                          gravity_factor=0.0)
        previous = float(rig.world.linear_velocity[crate][0])
        lost = 0.0
        for _ in range(120):
            rig.advance()
            now = float(rig.world.linear_velocity[crate][0])
            if abs(now - previous) > 1.0:
                lost = 20.0 * (previous - now)
                break
            previous = now
        assert lost < -50.0
        rig.character.knockback((lost, 0.0, 0.0))
        assert rig.character.push[0] == pytest.approx(lost / 80.0, abs=1e-9)
        rig.advance(3)
        assert float(rig.character.velocity[0]) < -1.0
        assert np.isfinite(rig.character.position).all()
