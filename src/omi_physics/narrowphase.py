"""Narrow phase — turn broad-phase pairs into contact manifolds.

Builds world-space proxies for each body's collider and dispatches the pair to
:mod:`omi_physics.collide`.  Proxies are cached per body per step so a
body in many pairs is transformed once.  When the compiled accelerator is
present, box↔box pairs -- the common case -- skip proxy construction entirely and
go straight to the native contact generator over batched rotation matrices.
"""
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np

from .body import make_proxy, Proxy
from . import collide
from .mathutil import quat_to_matrix

try:
    from . import _collide_native as _native
except ImportError:                            # pragma: no cover - pure-Python fallback
    _native = None

if TYPE_CHECKING:
    from .world import PhysicsWorld
    from .collide import Contact

_SPHERE = 1   # world._refit_kind codes
_BOX = 2


class NarrowPhase:
    """Turns broad-phase body-index pairs into contact manifolds, caching proxies per step."""

    def __init__(self) -> None:
        """Start with an empty proxy cache and no step recorded."""
        self._proxy_cache: Dict[int, tuple] = {}
        self._stamp = -1

    def _proxy(self, world: "PhysicsWorld", i: int) -> Proxy:
        """World-space collider proxy for body ``i``, rebuilt only when the step stamp changes."""
        entry = self._proxy_cache.get(i)
        if entry is None or entry[0] != self._stamp:
            shape = world.shapes[world.collider_shape[i]]
            proxy = make_proxy(shape, world.position[i], world.orientation[i])
            self._proxy_cache[i] = (self._stamp, proxy)
            return proxy
        return entry[1]

    def generate(self, world: "PhysicsWorld",
                 pairs: List[Tuple[int, int]]) -> "List[Contact]":
        """Contacts for every collidable pair; pairs missing a collider are skipped."""
        self._stamp += 1
        contacts: List[Contact] = []
        Contact = collide.Contact
        cshape = world.collider_shape

        if _native is None:
            for a, b in pairs:
                if cshape[a] < 0 or cshape[b] < 0:
                    continue
                contacts.extend(collide.collide(a, b, self._proxy(world, a),
                                                 self._proxy(world, b)))
            return contacts

        # Accelerated path: box↔box straight to native, everything else via proxies.
        n = world.body_count
        if getattr(world, '_refit_cache_n', -1) != n:
            world._build_refit_cache()
        kind = world._refit_kind
        half = world._refit_half
        pos = world.position
        rot = quat_to_matrix(world.orientation[:n])       # (n, 3, 3), batched
        box_box = _native.box_box
        sph_sph = _native.sphere_sphere
        sph_box = _native.sphere_box
        for a, b in pairs:
            if cshape[a] < 0 or cshape[b] < 0:
                continue
            ka, kb = kind[a], kind[b]
            if ka == _BOX and kb == _BOX:
                cnt, normal, pts, deps = box_box(pos[a], rot[a], half[a],
                                                 pos[b], rot[b], half[b])
            elif ka == _SPHERE and kb == _SPHERE:
                cnt, normal, pts, deps = sph_sph(pos[a], half[a, 0], pos[b], half[b, 0])
            elif ka == _SPHERE and kb == _BOX:
                cnt, normal, pts, deps = sph_box(pos[a], half[a, 0], pos[b], rot[b], half[b])
            elif ka == _BOX and kb == _SPHERE:
                # sphere is b: native returns the sphere→box normal (b→a); flip to a→b.
                cnt, normal, pts, deps = sph_box(pos[b], half[b, 0], pos[a], rot[a], half[a])
                normal = (-normal[0], -normal[1], -normal[2])
            else:
                contacts.extend(collide.collide(a, b, self._proxy(world, a),
                                                 self._proxy(world, b)))
                continue
            if cnt:
                nrm = np.array(normal)
                for pi in range(cnt):
                    contacts.append(Contact(a, b, np.array(pts[pi]), nrm, deps[pi]))
        return contacts
