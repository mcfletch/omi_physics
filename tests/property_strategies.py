"""Input strategies shared by the property suite.

The suite elsewhere checks the engine against scenes someone wrote down: a ball
dropped from three metres, a stack of five boxes, a wheel on a slope. Those say
the physics is right. The property files ask a different question -- whether the
invariants hold for inputs nobody thought of -- and this module is where the
inputs come from.

Degenerate shapes are deliberately in range: a zero-radius sphere, a box with a
flattened axis, a capsule with no mid-section, two colliders at the same point.
A game loading arbitrary glTF gets those from the asset pipeline, and what the
solver must never do is return a NaN, an infinite impulse, or a normal that is
not a direction.

Imported only after the calling module has established that hypothesis is
installed, so that an interpreter with no wheel for it runs the rest of the
suite (see any ``test_properties_*.py`` header).
"""
import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from omi_physics import mathutil, model
from omi_physics.body import BoxProxy, CapsuleProxy, ConvexProxy, SphereProxy

#: Generating a case is cheap and checking one is not, so the example counts are
#: modest and the deadline is off: these are a net for the cases nobody wrote
#: down, not a measurement. ``too_slow`` is suppressed because building a world
#: of bodies inside a strategy is slower than hypothesis expects a draw to be.
SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

#: For properties that step a whole world rather than call one routine.
STEPPING = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

#: Sizes a game actually ships, spanning six orders of magnitude: a bullet, a
#: crate, a landscape tile. Thresholds that are absolute rather than relative to
#: the input show up as a property that holds at one end and fails at the other.
SCALES = (1e-3, 1.0, 1e3)


def coordinates(limit: float = 1e3) -> st.SearchStrategy:
    """Finite world coordinates within ``limit`` of the origin.

    Subnormals are left out. They are a corner of IEEE arithmetic rather than
    one of the engine -- numpy signals ``underflow`` taking the length of a
    vector that holds one, before any of this code is reached -- and no scene
    positions anything at 5e-324 metres.
    """
    return st.floats(min_value=-limit, max_value=limit,
                     allow_nan=False, allow_infinity=False,
                     allow_subnormal=False)


def vectors(limit: float = 1e3) -> st.SearchStrategy:
    """A 3-vector of finite coordinates."""
    return st.lists(coordinates(limit), min_size=3, max_size=3).map(
        lambda xs: np.array(xs, dtype='d'))


def extents(low: float = 0.0, high: float = 10.0) -> st.SearchStrategy:
    """A half-extent or radius; ``low=0`` admits the flattened/zero-size shape."""
    return st.floats(min_value=low, max_value=high,
                     allow_nan=False, allow_infinity=False)


@st.composite
def directions(draw, limit: float = 1.0) -> np.ndarray:
    """A vector meant as a direction, the zero vector included.

    Components far below the largest are snapped to zero, which is what an
    axis-aligned ray is: a ground check points straight down rather than
    0.999999 down and a hundredth of a nanometre sideways. It also keeps the
    reciprocal the slab tests take finite, so a run's output is not filled with
    overflow warnings about a ray no scene would cast.
    """
    v = np.array(draw(st.lists(coordinates(limit), min_size=3, max_size=3)))
    biggest = float(np.max(np.abs(v)))
    if biggest:
        v[np.abs(v) < 1e-9 * biggest] = 0.0
    return v


@st.composite
def unit_quaternions(draw) -> np.ndarray:
    """A valid rotation quaternion, in ``(x, y, z, w)`` order.

    Built from an axis and an angle rather than drawn as four floats, so that
    every example *is* a rotation: a property about rotations should not spend
    its budget on 4-vectors that are not one. Angles reach past a half turn
    because the ones that break a quaternion routine are near 0 and near pi.
    """
    axis = draw(st.lists(coordinates(1.0), min_size=3, max_size=3))
    axis = np.array(axis, dtype='d')
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])
    angle = draw(st.floats(min_value=-2 * np.pi, max_value=2 * np.pi,
                           allow_nan=False, allow_infinity=False))
    return mathutil.quat_from_axis_angle(axis, angle)


def rough_quaternions() -> st.SearchStrategy:
    """A 4-vector long enough to normalise into a rotation, but not yet unit.

    What a caller loading a file hands in: a quaternion that has been through a
    float32 round trip, or scaled, or lerped between two others.
    """
    return st.lists(coordinates(4.0), min_size=4, max_size=4).filter(
        lambda q: np.linalg.norm(q) >= 1e-3).map(lambda q: np.array(q, dtype='d'))


def any_quaternions() -> st.SearchStrategy:
    """A 4-vector with no length guarantee at all, zero included."""
    return st.lists(coordinates(4.0), min_size=4, max_size=4).map(
        lambda q: np.array(q, dtype='d'))


@st.composite
def rotations(draw) -> np.ndarray:
    """A 3x3 world rotation matrix."""
    return mathutil.quat_to_matrix(draw(unit_quaternions()))


@st.composite
def sphere_proxies(draw, scale: float = 1.0, degenerate: bool = True):
    """A world-space sphere; ``degenerate`` admits a radius of zero."""
    return SphereProxy(draw(vectors(10.0 * scale)),
                       draw(extents(0.0 if degenerate else 1e-2, 2.0)) * scale)


@st.composite
def box_proxies(draw, scale: float = 1.0, degenerate: bool = True):
    """A world-space oriented box; ``degenerate`` admits a flattened axis."""
    low = 0.0 if degenerate else 1e-2
    half = np.array([draw(extents(low, 2.0)) for _ in range(3)]) * scale
    return BoxProxy(draw(vectors(10.0 * scale)), half, draw(rotations()))


@st.composite
def capsule_proxies(draw, scale: float = 1.0, degenerate: bool = True):
    """A world-space capsule; ``degenerate`` admits zero radius or no mid-section."""
    low = 0.0 if degenerate else 1e-2
    return CapsuleProxy(draw(vectors(10.0 * scale)),
                        draw(extents(low, 2.0)) * scale,
                        draw(extents(low, 2.0)) * scale,
                        draw(rotations()))


@st.composite
def convex_proxies(draw, scale: float = 1.0, degenerate: bool = True):
    """A world-space convex point cloud.

    ``degenerate`` lets the cloud collapse -- every point equal, or all of them
    on one line or plane -- which is what a hull cooked from a flat piece of
    level geometry looks like.
    """
    count = draw(st.integers(min_value=1 if degenerate else 4, max_value=10))
    points = np.array([draw(vectors(1.0)) for _ in range(count)]) * scale
    return ConvexProxy(points, draw(vectors(10.0 * scale)), draw(rotations()))


def support_proxies(scale: float = 1.0, degenerate: bool = True) -> st.SearchStrategy:
    """Any proxy the narrow phase can ask for a support point."""
    return st.one_of(
        sphere_proxies(scale, degenerate),
        box_proxies(scale, degenerate),
        capsule_proxies(scale, degenerate),
        convex_proxies(scale, degenerate),
    )


def analytic_proxies(scale: float = 1.0, degenerate: bool = True) -> st.SearchStrategy:
    """The pairs the narrow phase answers in closed form: spheres and boxes.

    Their contacts are exact, so the properties that would be approximate for
    GJK/EPA -- that the reported depth is the depth that separates them --
    can be asserted here as equalities.
    """
    return st.one_of(sphere_proxies(scale, degenerate), box_proxies(scale, degenerate))


def shapes(degenerate: bool = True) -> st.SearchStrategy:
    """A collider shape, as the model records it."""
    low = 0.0 if degenerate else 0.05
    return st.one_of(
        extents(low, 2.0).map(model.Shape.sphere),
        st.lists(extents(low, 3.0), min_size=3, max_size=3).map(
            lambda s: model.Shape.box(tuple(s))),
        st.tuples(extents(low, 2.0), extents(low, 1.0)).map(
            lambda hr: model.Shape.capsule(height=hr[0], radius=hr[1])),
    )


def motion_types() -> st.SearchStrategy:
    """One of the three OMI motion types."""
    return st.sampled_from([model.STATIC, model.KINEMATIC, model.DYNAMIC])


@st.composite
def world_recipes(draw, min_bodies: int = 1, max_bodies: int = 6,
                  ground: bool = True, filters: bool = False):
    """The description of a world, as plain data.

    Kept separate from the world it builds so a property can build the same
    scene twice -- which is how determinism is asked about, and it is a promise
    the CPU backend makes.
    """
    bodies = []
    for _ in range(draw(st.integers(min_value=min_bodies, max_value=max_bodies))):
        bodies.append({
            'shape': draw(shapes()),
            'type': draw(motion_types()),
            'mass': draw(st.floats(min_value=0.01, max_value=100.0)),
            'linear': tuple(draw(vectors(10.0))),
            'angular': tuple(draw(vectors(10.0))),
            'linear_damping': draw(extents(0.0, 1.0)),
            'angular_damping': draw(extents(0.0, 1.0)),
            'position': tuple(draw(vectors(3.0))),
            'orientation': tuple(draw(unit_quaternions())),
            'filter': draw(collision_filters()) if filters else None,
        })
    return {
        'gravity': draw(st.floats(min_value=0.0, max_value=30.0)),
        'friction': draw(extents(0.0, 1.5)),
        'restitution': draw(extents(0.0, 1.0)),
        'ground': ground,
        'bodies': bodies,
    }


def collision_filters() -> st.SearchStrategy:
    """A layer/mask filter, including the ones that exclude everything."""
    systems = st.lists(st.sampled_from(['world', 'player', 'debris']),
                       min_size=0, max_size=2, unique=True).map(tuple)
    return st.builds(model.CollisionFilter,
                     collisionSystems=systems,
                     collideWithSystems=st.one_of(st.none(), systems),
                     notCollideWithSystems=systems)


def build_world(recipe: dict):
    """The world a :func:`world_recipes` draw describes.

    Bodies land in a box a few metres across, so they start interpenetrating as
    often as not -- which is the state a level designer's first draft arrives
    in, and the one a solver has to recover from rather than diverge on.
    """
    from omi_physics.world import PhysicsWorld

    world = PhysicsWorld(
        gravity=model.Gravity(gravity=recipe['gravity'], direction=(0, -1, 0)),
        sleep_enabled=False)
    material = world.add_material(model.Material(
        staticFriction=recipe['friction'],
        dynamicFriction=recipe['friction'],
        restitution=recipe['restitution']))
    if recipe['ground']:
        floor = world.add_shape(model.Shape.box((40, 1, 40)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=floor, physicsMaterial=material),
                       position=(0, -0.5, 0))
    for body in recipe['bodies']:
        shape = world.add_shape(body['shape'])
        collider = model.Collider(shape=shape, physicsMaterial=material)
        if body['filter'] is not None:
            collider.collisionFilter = world.add_filter(body['filter'])
        world.add_body(
            model.Motion(type=body['type'], mass=body['mass'],
                         linearVelocity=body['linear'],
                         angularVelocity=body['angular'],
                         linearDamping=body['linear_damping'],
                         angularDamping=body['angular_damping']),
            collider=collider,
            position=body['position'],
            orientation=body['orientation'])
    return world


def worlds(min_bodies: int = 1, max_bodies: int = 6, ground: bool = True,
           filters: bool = False) -> st.SearchStrategy:
    """A world of randomly shaped, randomly placed, randomly moving bodies."""
    return world_recipes(min_bodies, max_bodies, ground, filters).map(build_world)



@st.composite
def triangles(draw, count: int = 4, scale: float = 1.0, degenerate: bool = True):
    """``(N, 3, 3)`` world-space triangles for the batched mesh tests.

    ``degenerate`` mixes in the triangles a real mesh is full of: a needle, a
    zero-area sliver, and one whose three vertices are the same point. They come
    out of exporters and decimators, they are the ones the batched separating-axis
    tests have no plane for, and the character controller walks over them.
    """
    verts = []
    for _ in range(count):
        a = draw(vectors(5.0)) * scale
        kind = draw(st.sampled_from(['proper', 'sliver', 'needle', 'point'])
                    if degenerate else st.just('proper'))
        if kind == 'point':
            verts.append(np.array([a, a, a]))
        elif kind == 'needle':
            b = a + draw(vectors(1.0)) * scale
            verts.append(np.array([a, b, a + (b - a) * 0.5]))
        elif kind == 'sliver':
            edge = draw(vectors(1.0)) * scale
            verts.append(np.array([a, a + edge, a + edge * (1.0 + 1e-9)]))
        else:
            verts.append(np.array([a, a + draw(vectors(2.0)) * scale,
                                   a + draw(vectors(2.0)) * scale]))
    return np.array(verts, dtype='d')


def translated(proxy, offset: np.ndarray):
    """The same proxy moved by ``offset``, for the translation-invariance checks."""
    offset = np.asarray(offset, dtype='d')
    if isinstance(proxy, SphereProxy):
        return SphereProxy(proxy.center + offset, proxy.radius)
    if isinstance(proxy, BoxProxy):
        return BoxProxy(proxy.center + offset, proxy.half, proxy.R)
    if isinstance(proxy, CapsuleProxy):
        return CapsuleProxy(proxy.center + offset, proxy.half_height,
                            proxy.radius, proxy.R)
    if isinstance(proxy, ConvexProxy):
        return ConvexProxy(proxy.local, proxy.center + offset, proxy.R)
    raise TypeError('no translation for %r' % type(proxy).__name__)


def shrunk(proxy, factor: float = 0.9):
    """The same proxy at ``factor`` of its size, about its own centre.

    A pair that still meets when both have been shrunk is meeting solidly rather
    than grazing -- which is the difference between a contact the solver acts on
    and one on a knife edge, where which way out is shorter comes down to the
    last bit of the coordinates.
    """
    if isinstance(proxy, SphereProxy):
        return SphereProxy(proxy.center, proxy.radius * factor)
    if isinstance(proxy, BoxProxy):
        return BoxProxy(proxy.center, proxy.half * factor, proxy.R)
    if isinstance(proxy, CapsuleProxy):
        return CapsuleProxy(proxy.center, proxy.half_height * factor,
                            proxy.radius * factor, proxy.R)
    if isinstance(proxy, ConvexProxy):
        return ConvexProxy(proxy.local * factor, proxy.center, proxy.R)
    raise TypeError('no shrinking for %r' % type(proxy).__name__)


def proxy_scale(proxy) -> float:
    """A length the proxy's own size, for tolerances that must not be absolute."""
    lo, hi = proxy.aabb()
    return max(float(np.max(hi - lo)), 1.0)
