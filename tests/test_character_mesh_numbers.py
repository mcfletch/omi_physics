"""A character among triangle meshes alone is collided with their triangles with numbers, and goes
exactly where the capsule proxy path takes it: on bumpy floors, against walls, up steps and
ramps, through jumps, slides and landings."""
import math

import numpy as np
import pytest

from omi_physics import character as character_module
from omi_physics import collide, model
from omi_physics.body import CapsuleProxy, make_proxy
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def box_triangles(low, high) -> tuple[np.ndarray, np.ndarray]:
    """The twelve triangles of an axis-aligned box from ``low`` to ``high``."""
    (x0, y0, z0), (x1, y1, z1) = low, high
    points = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype='d')
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return points, np.array([t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))])


def level(seed: int) -> model.Shape:
    """A bumpy floor with walls, steps, a ramp and a low ceiling, as one triangle mesh."""
    rng = np.random.default_rng(seed)
    size, cells = 40.0, 40
    xs = np.linspace(-size / 2, size / 2, cells + 1)
    gx, gz = np.meshgrid(xs, xs)
    heights = rng.uniform(0.0, 0.12) * np.sin(gx * rng.uniform(0.5, 2.0)) * np.cos(gz)
    points = [np.column_stack((gx.ravel(), heights.ravel(), gz.ravel()))]
    indices = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            indices += [(a, a + cells + 1, a + 1), (a + 1, a + cells + 1, a + cells + 2)]
    count = len(points[0])
    pieces = []
    for _ in range(10):                                   # walls and steps
        x, z = rng.uniform(-9.0, 9.0, 2)
        w, d = rng.uniform(0.3, 3.0, 2)
        h = float(rng.choice([0.15, 0.3, 2.5]))
        pieces.append(box_triangles((x, -0.5, z), (x + w, h, z + d)))
    pieces.append(box_triangles((-8.0, 1.6, 3.0), (-3.0, 2.0, 8.0)))       # a low ceiling
    ramp = np.array([(4.0, 0.0, -2.0), (8.0, 1.2, -2.0), (8.0, 1.2, 2.0), (4.0, 0.0, 2.0)])
    pieces.append((ramp, np.array([(0, 1, 2), (0, 2, 3)])))
    for piece_points, piece_indices in pieces:
        points.append(piece_points)
        indices += (np.asarray(piece_indices) + count).tolist()
        count += len(piece_points)
    return model.Shape.trimesh(np.vstack(points), np.array(indices))


def world_of(shape: model.Shape) -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(shape)))
    return world


def capabilities() -> CharacterCapabilities:
    return CharacterCapabilities(radius=0.3, standHeight=1.7, crouchHeight=0.9, runSpeed=6.0,
                                 stepHeight=0.35, jumpHeight=1.1, airControl=1.0,
                                 coyoteTime=0.0, jumpBuffer=0.0)


class _Proxied(character_module._Gather):  # noqa: SLF001 the gather under test
    """A gather that never offers its meshes as numbers: the capsule proxy path."""

    def __init__(self, *arguments, **named) -> None:
        super().__init__(*arguments, **named)
        self.meshes = None


def twins(seed: int, monkeypatch: pytest.MonkeyPatch) -> list[CharacterController]:
    shape = level(seed)
    pair = [CharacterController(world_of(shape), capabilities(), gravity=9.81) for _ in range(2)]
    for character in pair:
        character.safe_bind((0.5, 2.0, 0.5))
    return pair


def step_pair(pair, drive, step, monkeypatch) -> None:
    numbers, proxied = pair
    drive(numbers, step)
    numbers.update(DT)
    with monkeypatch.context() as patch:
        patch.setattr(character_module, '_Gather', _Proxied)
        drive(proxied, step)
        proxied.update(DT)


def roaming(seed: int):
    rng = np.random.default_rng(1000 + seed)
    turns = rng.uniform(-0.03, 0.03, size=4000)

    def drive(character, step):
        angle = 0.008 * step + float(turns[: step + 1].sum()) + seed
        mode = 'run' if step % 400 < 300 else 'walk'
        character.set_move((math.cos(angle), 0.0, math.sin(angle)), mode)
        if step % 170 == 60:
            character.jump()
        elif step % 230 == 120:
            character.slide(0.4, 0.2)
        elif step % 310 == 200:
            character.set_crouch(True)
        elif step % 310 == 260:
            character.set_crouch(False)
    return drive


@pytest.mark.parametrize('seed', range(12))
def test_a_character_among_meshes_goes_exactly_where_the_proxy_path_takes_it(seed, monkeypatch):
    pair = twins(seed, monkeypatch)
    drive = roaming(seed)
    grounded = 0
    for step in range(1200):
        step_pair(pair, drive, step, monkeypatch)
        numbers, proxied = pair
        assert numbers.position.tolist() == proxied.position.tolist(), step
        assert numbers.velocity.tolist() == proxied.velocity.tolist(), step
        assert numbers.grounded == proxied.grounded, step
        grounded += numbers.grounded
    assert grounded > 400


def test_a_contact_with_a_wall_is_found_with_numbers(monkeypatch):
    """Running into a wall stops the capsule at it: a contact missed would let it through."""
    points, indices = box_triangles((2.0, -0.5, -5.0), (2.4, 3.0, 5.0))
    floor, floor_indices = box_triangles((-10.0, -1.0, -10.0), (10.0, 0.0, 10.0))
    shape = model.Shape.trimesh(np.vstack((floor, points)), np.vstack((floor_indices, indices + 8)))
    character = CharacterController(world_of(shape), capabilities(), gravity=9.81)
    character.safe_bind((0.0, 1.0, 0.0))
    for _ in range(240):
        character.set_move((1.0, 0.0, 0.0), 'run')
        character.update(DT)
    assert 1.6 < character.position[0] < 2.0 - 0.3 + 1e-3
    assert character.grounded


@pytest.mark.skipif(collide._native is None,  # noqa: SLF001 whether the compiled module loaded
                    reason='the compiled accelerators are not in use')
def test_among_meshes_alone_depenetration_makes_no_capsule_proxy(monkeypatch):
    character = CharacterController(world_of(level(3)), capabilities(), gravity=9.81)
    character.safe_bind((0.5, 2.0, 0.5))
    inside = []
    made = []
    push_out = CharacterController._push_out  # noqa: SLF001 the depenetration is what is counted

    def counted(self, *arguments, **named):
        inside.append(True)
        try:
            return push_out(self, *arguments, **named)
        finally:
            inside.pop()
    monkeypatch.setattr(CharacterController, '_push_out', counted)
    original = CapsuleProxy.__init__
    monkeypatch.setattr(
        CapsuleProxy, '__init__',
        lambda self, *a, **k: (inside and made.append(a)) or original(self, *a, **k))
    monkeypatch.setattr(collide, 'capsule_mesh_pushes',
                        lambda *a, **k: pytest.fail('the proxy path was taken'))
    pushes = []
    numbers = collide.upright_capsule_mesh_pushes
    monkeypatch.setattr(collide, 'upright_capsule_mesh_pushes',
                        lambda *a: pushes.append(a) or numbers(*a))
    drive = roaming(3)
    for step in range(200):
        drive(character, step)
        character.update(DT)
    assert pushes
    assert made == []


@pytest.mark.parametrize('seed', range(20))
def test_the_numbers_are_the_proxy_pushes_to_the_bit(seed):
    rng = np.random.default_rng(seed)
    mesh = make_proxy(level(seed), np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    for _ in range(40):
        x, z = rng.uniform(-10.0, 10.0, 2)
        y = float(rng.uniform(-0.2, 2.5))
        half, radius = float(rng.uniform(0.1, 0.8)), float(rng.uniform(0.1, 0.5))
        capsule = CapsuleProxy(np.array([x, y, z]), half, radius, np.eye(3))
        verts = mesh.candidate_vertices(*capsule.aabb())
        pushes, depths = collide.capsule_mesh_pushes(capsule, mesh, verts)
        wanted = [(tuple(push.tolist()), float(depth)) for push, depth in zip(pushes, depths,
                                                                              strict=True)]
        got = [(tuple(float(v) for v in push), float(depth)) for push, depth in
               collide.upright_capsule_mesh_pushes(verts, x, y, z, half, radius)]
        assert got == wanted
