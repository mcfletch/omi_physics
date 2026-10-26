"""A step in a world of many static bodies looks only at the ones that changed: none is compared
with its last pose unless it was placed, added or removed, and each way of changing one is seen
by the next step's contacts."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.mathutil import quat_from_axis_angle
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
STATICS = 100_000


def forest(count: int = STATICS) -> tuple[PhysicsWorld, int]:
    """``count`` posts and a floor, and a ball rolling across the floor between them."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    post = world.add_shape(model.Shape.box((0.4, 2.0, 0.4)))
    rng = np.random.default_rng(3)
    spots = np.column_stack([rng.uniform(-400, 400, count), np.full(count, 1.0),
                             rng.uniform(-400, 400, count)])
    spots = spots[np.hypot(spots[:, 0], spots[:, 2]) > 20.0]
    world.add_bodies(post, spots, collider=model.Collider(shape=post))
    floor = world.add_shape(model.Shape.box((1000.0, 1.0, 1000.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    ball = world.add_shape(model.Shape.sphere(0.3))
    rolling = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0, linearVelocity=(2, 0, 0)),
                             collider=model.Collider(shape=ball), position=(0.0, 0.3, 0.0))
    return world, rolling


def test_a_step_compares_no_static_body_with_its_last_pose(monkeypatch):
    monkeypatch.delenv('OMI_PHYSICS_CHECK_STATICS', raising=False)
    world, _ball = forest()
    for _ in range(3):
        world.step(DT)
    compared = []
    original = np.array_equal

    def counting(a, b, *rest, **named):
        if np.size(a) >= STATICS // 2:
            compared.append(np.shape(a))
        return original(a, b, *rest, **named)

    monkeypatch.setattr(np, 'array_equal', counting)
    for _ in range(20):
        world.step(DT)
    assert compared == []


def ball_on_a_pad() -> tuple[PhysicsWorld, int, int]:
    """A ball resting on a thin pad over nothing, among enough posts for the index of them."""
    world, _rolling = forest(2000)
    pad = world.add_shape(model.Shape.box((4.0, 0.2, 0.4)))
    under = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=pad),
                           position=(100.0, 50.0, 100.0))
    ball = world.add_shape(model.Shape.sphere(0.3))
    resting = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                             collider=model.Collider(shape=ball), position=(101.5, 50.4, 100.0))
    for _ in range(60):
        world.step(DT)
    assert float(world.position[resting][1]) == pytest.approx(50.4, abs=0.02)
    return world, under, resting


def place(world, pad):
    world.place_body(pad, position=(130.0, 50.0, 100.0))


def turn(world, pad):
    world.place_body(pad, orientation=quat_from_axis_angle((0.0, 1.0, 0.0), np.pi / 2))


def write(world, pad):
    world.position[pad] = (130.0, 50.0, 100.0)
    world.moved([pad])


def refit(world, pad):
    world.position[pad] = (130.0, 50.0, 100.0)
    world.refit_aabbs(rows=np.array([pad]))


def remove(world, pad):
    world.remove_body(pad)


@pytest.mark.parametrize('change', [place, turn, write, refit, remove])
def test_a_pad_moved_from_under_a_ball_lets_it_fall_on_the_next_step(change):
    world, pad, ball = ball_on_a_pad()
    change(world, pad)
    for _ in range(90):
        world.step(DT)
    assert float(world.position[ball][1]) < 49.5


@pytest.mark.parametrize('many', [False, True])
def test_a_pad_added_under_a_falling_ball_catches_it(many):
    world, pad, ball = ball_on_a_pad()
    world.remove_body(pad)
    world.step(DT)
    shape = world.add_shape(model.Shape.box((4.0, 0.2, 4.0)))
    if many:
        world.add_bodies(shape, [(101.5, 48.0, 100.0)], collider=model.Collider(shape=shape))
    else:
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                       position=(101.5, 48.0, 100.0))
    for _ in range(120):
        world.step(DT)
    assert float(world.position[ball][1]) == pytest.approx(48.4, abs=0.03)


def test_with_the_statics_checked_a_bare_write_is_refused(monkeypatch):
    monkeypatch.setenv('OMI_PHYSICS_CHECK_STATICS', '1')
    world, pad, _ball = ball_on_a_pad()
    world.position[pad] = (130.0, 50.0, 100.0)
    with pytest.raises(RuntimeError, match='moved'):
        world.step(DT)


def test_with_the_statics_checked_a_marked_write_is_taken(monkeypatch):
    monkeypatch.setenv('OMI_PHYSICS_CHECK_STATICS', '1')
    world, pad, ball = ball_on_a_pad()
    write(world, pad)
    for _ in range(90):
        world.step(DT)
    assert float(world.position[ball][1]) < 49.5
