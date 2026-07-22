"""The OMI physics data model, natively.

These are plain data structures (dataclasses) mirroring the OMI glTF physics
extension family field-for-field, with the spec defaults.  The loader, the
scenegraph nodes, and :class:`~omi_physics.world.PhysicsWorld` all
speak this same structure, so glTF import/export is a near-identity mapping and
there is no private format to keep in sync.

References:
    OMI_physics_shape / OMI_physics_body / OMI_physics_gravity / OMI_physics_joint
    https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0
"""
from dataclasses import dataclass, field
from typing import List, Optional
from numpy.typing import ArrayLike
import numpy as np

from .mathutil import Vec

# Motion types (OMI_physics_body.motion.type)
STATIC = 'static'
KINEMATIC = 'kinematic'
DYNAMIC = 'dynamic'

# Combine modes (OMI physics material)
AVERAGE = 'average'
MINIMUM = 'minimum'
MAXIMUM = 'maximum'
MULTIPLY = 'multiply'


@dataclass
class Shape:
    """An ``OMI_physics_shape`` entry: one collision proxy, shared by index.

    Only the fields relevant to ``type`` are meaningful; the rest keep their
    defaults.  ``convex``/``trimesh`` carry an explicit vertex array (and, for
    ``trimesh``, triangle indices) so the simulation never depends on render
    geometry.
    """
    type: str = 'box'
    size: tuple = (1.0, 1.0, 1.0)              # box
    radius: float = 0.5                         # sphere
    height: float = 2.0                         # capsule (mid) / cylinder (total)
    radiusBottom: float = 0.5                   # capsule / cylinder
    radiusTop: float = 0.5                      # capsule / cylinder
    points: Optional[np.ndarray] = None         # convex / trimesh vertices (M,3)
    indices: Optional[np.ndarray] = None        # trimesh triangle indices (T,3)
    mesh: int = -1                              # glTF mesh index (convex / trimesh)

    @classmethod
    def box(cls, size: Vec = (1.0, 1.0, 1.0)) -> 'Shape':
        """A box collider with full extents ``size`` (x, y, z)."""
        return cls(type='box', size=tuple(size))

    @classmethod
    def sphere(cls, radius: float = 0.5) -> 'Shape':
        """A sphere collider of the given radius."""
        return cls(type='sphere', radius=float(radius))

    @classmethod
    def capsule(cls, height: float = 1.0, radius: float = 0.5) -> 'Shape':
        """A capsule collider: ``height`` is the cylindrical mid-section length."""
        return cls(type='capsule', height=float(height),
                   radiusBottom=float(radius), radiusTop=float(radius))

    @classmethod
    def cylinder(cls, height: float = 2.0, radius: float = 0.5) -> 'Shape':
        """A cylinder collider: ``height`` is the total length end to end."""
        return cls(type='cylinder', height=float(height),
                   radiusBottom=float(radius), radiusTop=float(radius))

    @classmethod
    def convex(cls, points: ArrayLike) -> 'Shape':
        """A convex-hull collider from an ``(M, 3)`` local vertex cloud."""
        return cls(type='convex', points=np.asarray(points, dtype='d'))

    @classmethod
    def trimesh(cls, points: ArrayLike, indices: ArrayLike) -> 'Shape':
        """A static triangle-soup collider from vertices and ``(T, 3)`` triangle indices."""
        return cls(type='trimesh', points=np.asarray(points, dtype='d'),
                   indices=np.asarray(indices, dtype='i'))


@dataclass
class Motion:
    """``OMI_physics_body.motion`` — the rigid-body dynamics parameters.

    ``type: "static"`` is an immobile body (the Earth); ``mass`` times the
    gravity magnitude is its weight; ``centerOfMass`` is its centre of gravity.
    """
    type: str = DYNAMIC
    mass: float = 1.0
    centerOfMass: tuple = (0.0, 0.0, 0.0)
    inertiaDiagonal: tuple = (0.0, 0.0, 0.0)    # 0 => auto-derive from shape+mass
    inertiaOrientation: tuple = (0.0, 0.0, 0.0, 1.0)
    linearVelocity: tuple = (0.0, 0.0, 0.0)
    angularVelocity: tuple = (0.0, 0.0, 0.0)
    gravityFactor: float = 1.0
    linearDamping: float = 0.0
    angularDamping: float = 0.0
    quadraticDrag: float = 0.0                   # non-OMI; drag accel = -k|v|v


@dataclass
class Collider:
    """``OMI_physics_body.collider`` — a solid shape that generates contacts."""
    shape: int = -1                             # index into world shapes[]
    physicsMaterial: int = -1                   # index, -1 = default
    collisionFilter: int = -1                   # index, -1 = default


@dataclass
class Trigger:
    """``OMI_physics_body.trigger`` — a sensor shape (overlap events, no impulse)."""
    shape: int = -1
    collisionFilter: int = -1


@dataclass
class Material:
    """``OMI_physics_body`` ``physicsMaterials[]`` entry."""
    staticFriction: float = 0.6
    dynamicFriction: float = 0.6
    restitution: float = 0.0
    frictionCombine: str = AVERAGE
    restitutionCombine: str = AVERAGE


@dataclass
class CollisionFilter:
    """``collisionFilters[]`` — the standard layer/mask system.

    A body is a member of every system in ``collisionSystems`` and collides with
    a candidate only when their systems intersect the ``collideWithSystems`` /
    ``notCollideWithSystems`` policy.  ``collideWithSystems=None`` means "collide
    with everything not explicitly excluded".
    """
    collisionSystems: tuple = ()
    collideWithSystems: Optional[tuple] = None
    notCollideWithSystems: tuple = ()

    def collides_with(self, other: 'CollisionFilter') -> bool:
        """True if this filter and ``other`` may collide (symmetric: both must agree)."""
        return _one_way(self, other) and _one_way(other, self)


def _one_way(a: 'CollisionFilter', b: 'CollisionFilter') -> bool:
    """True if ``a``'s collide/exclude policy admits a body in ``b``'s systems."""
    b_systems = set(b.collisionSystems)
    if b_systems & set(a.notCollideWithSystems):
        return False
    if a.collideWithSystems is None:
        return True
    return bool(b_systems & set(a.collideWithSystems))


# Gravity volume types (OMI_physics_gravity)
DIRECTIONAL = 'directional'
POINT = 'point'


@dataclass
class Gravity:
    """``OMI_physics_gravity`` — global gravity or a per-volume gravity zone.

    Global (document-level) gravity is a uniform directional field: ``gravity``
    magnitude (m/s²) along ``direction``.  A volume additionally carries
    ``priority`` / ``replace`` / ``stop`` and, for ``type="point"``, pulls toward
    the body's world ``center``.
    """
    type: str = DIRECTIONAL
    gravity: float = 9.81
    direction: tuple = (0.0, -1.0, 0.0)
    priority: int = 0
    replace: bool = False
    stop: bool = False
    center: tuple = (0.0, 0.0, 0.0)             # world centre for point gravity


@dataclass
class JointLimit:
    """One ``OMI_physics_joint`` limit row over a set of axes."""
    linearAxes: tuple = ()                      # subset of (0,1,2)
    angularAxes: tuple = ()
    min: float = -np.inf
    max: float = np.inf
    stiffness: float = np.inf
    damping: float = 0.0


@dataclass
class JointDrive:
    """One ``OMI_physics_joint`` drive (motor/spring) row."""
    type: str = 'linear'                        # 'linear' | 'angular'
    mode: str = 'force'                          # 'force' | 'acceleration'
    axis: int = 0
    maxForce: float = np.inf
    positionTarget: Optional[float] = None
    velocityTarget: Optional[float] = None
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class Joint:
    """A ``physicsJoints[]`` entry: limits + drives between two bodies."""
    limits: List[JointLimit] = field(default_factory=list)
    drives: List[JointDrive] = field(default_factory=list)


@dataclass
class JointAttach:
    """Per-body joint attachment (``OMI_physics_body`` joint)."""
    joint: int = -1                             # index into physicsJoints[]
    connectedNode: int = -1                     # body index of the other end
    enableCollision: bool = False


DEFAULT_MATERIAL = Material()
DEFAULT_FILTER = CollisionFilter()

MATERIAL_PRESETS = {
    'wood':   Material(staticFriction=0.5, dynamicFriction=0.4, restitution=0.2),
    'metal':  Material(staticFriction=0.4, dynamicFriction=0.3, restitution=0.1),
    'rubber': Material(staticFriction=1.0, dynamicFriction=0.9, restitution=0.8),
    'ice':    Material(staticFriction=0.05, dynamicFriction=0.02, restitution=0.05),
}


def combine(a: float, b: float, mode: str) -> float:
    """Combine two materials' friction/restitution per an OMI combine ``mode``.

    ``mode`` is one of :data:`AVERAGE` (the default for any unrecognised value),
    :data:`MINIMUM`, :data:`MAXIMUM`, or :data:`MULTIPLY`.
    """
    if mode == MINIMUM:
        return min(a, b)
    if mode == MAXIMUM:
        return max(a, b)
    if mode == MULTIPLY:
        return a * b
    return 0.5 * (a + b)
