"""A character among height fields alone is collided with them with numbers, and goes where the
contact solver's answer takes it: the same pose, step for step, on a grid, on a function ground,
up a slope, through a jump, a slide and a landing, and beside a body that moves."""
import math

import numpy as np
import pytest

from omi_physics import collide, heightfield, model
from omi_physics.body import CapsuleProxy, HeightFieldProxy, make_proxy
from omi_physics.character import IDENT, CharacterCapabilities, CharacterController
from omi_physics.mathutil import quat_from_axis_angle, quat_to_matrix
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
#: Metres two poses of the same step may differ by.
AGREE = 1e-9


def bumpy_heights(rows: int = 161, columns: int = 161, spacing: float = 0.5) -> np.ndarray:
    z, x = np.mgrid[0:rows, 0:columns] * spacing
    return 0.6 * np.sin(x * 0.31) * np.cos(z * 0.23) + 0.08 * x + 0.3 * np.sin(z * 1.7)


def bumpy_grid() -> heightfield.GridSurface:
    return heightfield.GridSurface(bumpy_heights(), origin=(-40.0, -2.0, -40.0), spacing=0.5)


def rolling(x, z):
    return 0.5 * np.sin(np.asarray(x) * 0.4) + 0.25 * np.cos(np.asarray(z) * 0.3)


def world_on(shape: model.Shape, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)
             ) -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(shape)),
                   position=position, orientation=orientation)
    return world


def gentle_heights(rows: int = 161, columns: int = 161, spacing: float = 0.5) -> np.ndarray:
    z, x = np.mgrid[0:rows, 0:columns] * spacing
    return 0.4 * np.sin(x * 0.2) * np.cos(z * 0.15) + 0.12 * x + 0.02 * np.sin(z * 2.9)


def grid_world() -> PhysicsWorld:
    return world_on(model.Shape.heightfield(bumpy_heights(), origin=(-40.0, -2.0, -40.0),
                                            spacing=0.5))


def gentle_world() -> PhysicsWorld:
    """A grid with a cliff at one edge: gentle where the character runs, steep as a whole."""
    heights = gentle_heights()
    heights[:, -3:] += 6.0
    return world_on(model.Shape.heightfield(heights, origin=(-40.0, -2.0, -40.0), spacing=0.5))


def function_world() -> PhysicsWorld:
    return world_on(model.Shape.heightfield(sample=rolling, vectorized=True,
                                            height_range=(-1.0, 1.0)))


def slope_world(rise: float) -> PhysicsWorld:
    return world_on(model.Shape.heightfield(
        sample=lambda x, z: rise * np.asarray(x, dtype='d') + 0.0 * np.asarray(z, dtype='d'),
        vectorized=True, extent=((-80.0, -80.0), (80.0, 80.0)),
        height_range=(-80.0 * rise - 1.0, 80.0 * rise + 1.0)))


# -- the contacts ------------------------------------------------------------------------

def pushes_by_collide(field: HeightFieldProxy, centre, half: float, radius: float):
    capsule = CapsuleProxy(np.asarray(centre, dtype='d'), half, radius, quat_to_matrix(IDENT))
    return [(tuple((-c.normal).tolist()), c.depth) for c in collide.collide(0, 1, capsule, field)]


FIELDS = {
    'grid': lambda: HeightFieldProxy(bumpy_grid(), (0.0, 0.0, 0.0), np.eye(3)),
    'grid moved': lambda: HeightFieldProxy(bumpy_grid(), (3.25, 1.5, -2.75), np.eye(3)),
    'grid turned': lambda: HeightFieldProxy(
        bumpy_grid(), (1.0, 0.0, 2.0), quat_to_matrix(quat_from_axis_angle((0.0, 1.0, 0.0), 0.4))),
    'function': lambda: make_proxy(model.Shape.heightfield(sample=rolling, vectorized=True),
                                   (0.0, 0.0, 0.0), IDENT),
    'function, one point at a time': lambda: make_proxy(model.Shape.heightfield(
        sample=lambda x, z: 0.3 * math.sin(x) * math.cos(0.7 * z)), (0.5, -0.25, 0.0), IDENT),
}


@pytest.mark.parametrize('name', sorted(FIELDS))
@pytest.mark.parametrize('seed', range(4))
def test_an_upright_capsule_pushes_out_of_a_height_field_as_collide_has_it(name, seed):
    """To a nanometre: a sphere left out by the local bound changes when the others stop sliding."""
    field = FIELDS[name]()
    rng = np.random.default_rng(seed)
    found = 0
    for _ in range(60):
        x, z = rng.uniform(-15.0, 15.0, 2)
        ground = float(field.surface.heights(np.array([x - field.center[0]]),
                                             np.array([z - field.center[2]]))[0]) + field.center[1]
        radius = float(rng.choice([0.15, 0.3]))
        half = float(rng.choice([0.2, 0.6, 1.0]))
        # Resting, sunk a little, buried to the shoulders, and clear of the ground.
        y = ground + half + radius + float(rng.choice([-0.02, -0.3, -1.5, 0.4]))
        wanted = pushes_by_collide(field, (x, y, z), half, radius)
        got = collide.upright_capsule_heightfield_pushes(field, x, y, z, half, radius)
        assert len(got) == len(wanted)
        for (push, depth), (push_wanted, depth_wanted) in zip(got, wanted, strict=True):
            assert push == pytest.approx(push_wanted, abs=1e-9)
            assert depth == pytest.approx(depth_wanted, abs=1e-9)
        found += len(got)
    assert found


@pytest.mark.parametrize('seed', range(6))
def test_a_sphere_above_its_local_clearance_never_touches_the_ground(seed):
    """The bound :meth:`GridSurface.column_contacts` leaves spheres out by: none it would leave out
    touches, on grids of every slope up to the gentlest bound and beyond."""
    rng = np.random.default_rng(seed)
    rows = columns = 60
    z, x = np.mgrid[0:rows, 0:columns] * 0.5
    tilt = rng.uniform(0.0, 0.3)
    heights = (tilt * x * rng.uniform(-1, 1) + tilt * z * rng.uniform(-1, 1)
               + rng.uniform(0.0, 0.15) * np.sin(x * rng.uniform(0.5, 3.0) + z))
    surface = heightfield.GridSurface(heights, spacing=0.5)
    checked = 0
    for _ in range(400):
        px, pz = rng.uniform(4.0, 25.0, 2)
        radius = float(rng.uniform(0.05, 0.5))
        limit, top = surface._local_clearance(px, pz, radius)  # noqa: SLF001 the bound under test
        if not math.isfinite(limit):
            continue
        ground = surface.height_at(px, pz)
        for height in np.linspace(limit, min(top, 10.0), 7):
            touching, *_ = surface.sphere_contact(px, ground + float(height), pz, radius)
            assert not touching
            checked += 1
    assert checked


def test_on_gentle_ground_a_standing_capsule_slides_one_sphere(monkeypatch):
    # The Python column; the compiled one slides none in Python (test_heightfield_column_native).
    monkeypatch.setattr(heightfield, '_native', None)
    heights = gentle_heights()
    heights[:, -3:] += 6.0
    surface = heightfield.GridSurface(heights, origin=(-40.0, -2.0, -40.0), spacing=0.5)
    assert surface.lipschitz > 5.0
    slides = []
    original = surface.sphere_contact
    surface.sphere_contact = lambda *a: slides.append(a) or original(*a)
    ground = surface.height_at(1.0, 2.0)
    heights_up = [ground + 0.15 - 0.01 + 0.17 * k for k in range(9)]
    assert surface.column_contacts(1.0, 2.0, heights_up, 0.15)
    assert len(slides) == 1


def test_a_buried_capsule_keeps_the_most_spread_contacts_collide_keeps():
    field = FIELDS['grid']()
    x, z = 4.0, 6.0
    ground = field.surface.height_at(x, z)
    heights = [ground - 2.0 + 0.4 * k for k in range(8)]
    assert len(field.surface.column_contacts(x, z, heights, 0.3)) > 4
    wanted = pushes_by_collide(field, (x, ground - 1.0, z), 1.0, 0.3)
    assert 1 < len(wanted) <= 4
    assert collide.upright_capsule_heightfield_pushes(field, x, ground - 1.0, z, 1.0, 0.3) == wanted


# -- the character -----------------------------------------------------------------------

def capabilities() -> CharacterCapabilities:
    return CharacterCapabilities(radius=0.15, standHeight=1.7, crouchHeight=0.9, runSpeed=8.0,
                                 jumpHeight=1.1, airControl=1.0, coyoteTime=0.0, jumpBuffer=0.0)


def twins(make_world, start=(0.0, 4.0, 0.0)):
    """The same character in two worlds of its own: collided with numbers, and through collide."""
    pair = []
    for fast in (True, False):
        character = CharacterController(make_world(), capabilities(), gravity=9.81)
        character.ground_fast_path = fast
        character.safe_bind(start)
        pair.append(character)
    return pair


def run_together(pair, steps: int, drive) -> int:
    """Step both characters, driving each the same way; returns the steps either was grounded in."""
    grounded = 0
    for step in range(steps):
        for character in pair:
            drive(character, step)
            character.update(DT)
        fast, slow = pair
        assert np.max(np.abs(fast.position - slow.position)) <= AGREE, step
        assert fast.grounded == slow.grounded, step
        assert np.max(np.abs(fast.velocity - slow.velocity)) <= AGREE / DT, step
        grounded += fast.grounded
    return grounded


def circling(character, step):
    angle = step * 0.01
    character.set_move((math.cos(angle), 0.0, math.sin(angle)), 'run')


GROUNDS = [grid_world, gentle_world, function_world]
GROUND_NAMES = ['grid', 'gentle grid', 'function']


@pytest.mark.parametrize('make_world', GROUNDS, ids=GROUND_NAMES)
def test_a_character_running_on_the_ground_goes_where_collide_takes_it(make_world):
    pair = twins(make_world)
    assert run_together(pair, 1000, circling) > 900


@pytest.mark.parametrize('rise', [0.2, 0.7, 1.6])
def test_a_character_on_a_slope_goes_where_collide_takes_it(rise):
    """Up a walkable slope, up one at the limit, and on one too steep to stand on, sliding."""
    pair = twins(lambda: slope_world(rise), start=(0.0, 4.0, 0.0))

    def uphill(character, step):
        character.set_move((1.0, 0.0, 0.3 * math.sin(step * 0.02)), 'run')

    run_together(pair, 1000, uphill)


@pytest.mark.parametrize('make_world', GROUNDS, ids=GROUND_NAMES)
def test_jumps_slides_and_landings_go_where_collide_takes_them(make_world):
    pair = twins(make_world, start=(0.0, 6.0, 0.0))
    landings = [0, 0]

    def playing(character, step):
        circling(character, step)
        if step % 150 == 40:
            character.jump()
        elif step % 150 == 110:
            character.slide(0.5, 0.2, steer=step % 300 == 110)

    for step in range(1000):
        for index, character in enumerate(pair):
            playing(character, step)
            character.update(DT)
            landings[index] += character.landing is not None
        assert np.max(np.abs(pair[0].position - pair[1].position)) <= AGREE, step
    assert landings[0] == landings[1] >= 6


def test_a_fall_from_a_height_lands_where_collide_lands_it():
    pair = twins(grid_world, start=(2.0, 25.0, -3.0))
    run_together(pair, 600, lambda character, step: None)
    assert pair[0].grounded


def test_beside_a_body_that_moves_the_character_is_collided_through_collide_and_agrees(monkeypatch):
    """A crate near the capsule turns the numbers off; the poses still agree."""
    pair = []
    for fast in (True, False):
        world = grid_world()
        crate = world.add_shape(model.Shape.box((0.6, 0.6, 0.6)))
        world.add_body(model.Motion(type=model.DYNAMIC, mass=30.0),
                       collider=model.Collider(shape=crate), position=(1.5, 2.5, 0.0))
        character = CharacterController(world, capabilities(), gravity=9.81)
        character.ground_fast_path = fast
        character.safe_bind((0.0, 4.0, 0.0))
        pair.append(character)
    calls = []
    original = collide.collide
    monkeypatch.setattr(collide, 'collide', lambda *a: calls.append(a) or original(*a))
    run_together(pair, 200, lambda character, step: character.set_move((1.0, 0.0, 0.0), 'walk'))
    assert calls


def test_among_grounds_alone_the_contact_solver_is_not_asked(monkeypatch):
    character = CharacterController(grid_world(), capabilities(), gravity=9.81)
    character.safe_bind((0.0, 4.0, 0.0))
    calls = []
    original = collide.collide
    monkeypatch.setattr(collide, 'collide', lambda *a: calls.append(a) or original(*a))
    for step in range(240):
        circling(character, step)
        if step == 60:
            character.jump()
        if step == 150:
            character.slide(0.6, 0.3)
        character.update(DT)
    assert character.grounded
    assert calls == []
