"""A ray parallel to the plane of a triangle: a miss, with no floating-point error on the way."""
import numpy as np

from omi_physics import raycast as raycast_module

#: A wall standing in the plane z = 0.
WALL = np.array([[(0.0, 0.0, 0.0), (0.0, 2.5, 0.0), (4.0, 2.5, 0.0)],
                 [(0.0, 0.0, 0.0), (4.0, 2.5, 0.0), (4.0, 0.0, 0.0)]])
#: A floor under it, facing up.
FLOOR = np.array([[(-5.0, 0.0, -5.0), (5.0, 0.0, 5.0), (5.0, 0.0, -5.0)]])
DOWN = np.array([0.0, -1.0, 0.0])


def test_a_ray_down_beside_a_wall_misses_it_with_errors_raised():
    with np.errstate(all='raise'):
        found = raycast_module._hit_triangles(  # noqa: SLF001 the arithmetic is what is tested
            np.array([1.0, 1.3, 0.2]), DOWN, WALL, 0.8)
    assert found is None


def test_the_floor_behind_a_wall_seen_edge_on_is_still_found():
    triangles = np.concatenate([WALL, FLOOR])
    with np.errstate(all='raise'):
        found = raycast_module._hit_triangles(  # noqa: SLF001 the arithmetic is what is tested
            np.array([1.0, 0.3, 0.2]), DOWN, triangles, 0.8)
    assert found is not None
    distance, normal, index = found
    assert distance == 0.3
    assert index == 2
    assert abs(normal[1]) == 1.0
