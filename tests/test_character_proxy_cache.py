"""Static collision proxies belong to the *world*, not to each avatar.

A trimesh proxy is not a handle: it transforms every vertex into world space
and builds a uniform grid over the result. One per character controller means
one copy of the level per person walking in it, which is a level's worth of
memory and a level's worth of grid-building for each opponent that joins —
paid the first time each of them takes a step, so it shows up as a hitch at
the start of a fight rather than as a number anybody profiled.

They never move, which is what makes sharing them correct.
"""

from __future__ import annotations

import numpy as np

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld


def world():
    return PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))


def floor(w, y=0.0, reach=20.0):
    points = np.array([(-reach, y, -reach), (reach, y, -reach),
                       (reach, y, reach), (-reach, y, reach)], dtype='d')
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = w.add_shape(model.Shape.trimesh(points, indices))
    return w.add_body(model.Motion(type=model.STATIC),
                      collider=model.Collider(shape=shape), position=(0, 0, 0))


def walker(w, position=(0.0, 1.0, 0.0)):
    return CharacterController(w, CharacterCapabilities(), position=position)


def test_two_avatars_in_one_world_share_a_static_proxy():
    w = world()
    body = floor(w)
    one, two = walker(w), walker(w, (2.0, 1.0, 0.0))
    assert one._static_proxy(body) is two._static_proxy(body)


def test_avatars_in_different_worlds_do_not():
    """A proxy holds the geometry in world space; two worlds are two levels."""
    first, second = world(), world()
    body = floor(first)
    assert floor(second) == body
    assert walker(first)._static_proxy(body) \
        is not walker(second)._static_proxy(body)


def test_the_shared_cache_goes_when_the_world_does():
    """Held weakly, so an unloaded map takes its collision copy with it."""
    import gc
    from omi_physics import character
    w = world()
    floor(w)
    walker(w).update(1.0 / 60.0)
    assert w in character._STATIC_PROXIES
    del w
    gc.collect()
    assert len(character._STATIC_PROXIES) == 0


def test_sharing_does_not_change_where_anybody_stands():
    """The saving has to be invisible, which is the only reason it is safe.

    Two avatars dropped onto the same floor from the same height land in the
    same place, whether or not they read the geometry through one proxy.
    """
    w = world()
    floor(w)
    one, two = walker(w, (0.0, 3.0, 0.0)), walker(w, (0.0, 3.0, 0.0))
    for _ in range(120):
        one.update(1.0 / 60.0)
        two.update(1.0 / 60.0)
    assert one.base()[1] == two.base()[1]
    assert one.grounded and one.base()[1] < 0.01
