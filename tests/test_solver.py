"""Phase 1 solver: restitution, resting stacks, friction, determinism (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def ground_world(gravity_dir=(0, -1, 0), g=9.81, restitution=0.0, friction=0.6):
    world = PhysicsWorld(gravity=model.Gravity(gravity=g, direction=gravity_dir),
                         sleep_enabled=False)
    mat = world.add_material(model.Material(staticFriction=friction,
                                            dynamicFriction=friction,
                                            restitution=restitution))
    ground_shape = world.add_shape(model.Shape.box((40, 1, 40)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground_shape, physicsMaterial=mat),
                   position=(0, -0.5, 0))
    return world, mat


def drop_sphere(world, mat, height, radius=0.5):
    s = world.add_shape(model.Shape.sphere(radius))
    return world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=s, physicsMaterial=mat),
                          position=(0, height, 0))


def test_no_restitution_comes_to_rest():
    world, mat = ground_world(restitution=0.0)
    body = drop_sphere(world, mat, height=3.0)
    for _ in range(600):
        world.step(DT)
    assert world.position[body][1] == pytest.approx(0.5, abs=0.05)   # radius above ground top (0)
    assert abs(world.linear_velocity[body][1]) < 0.2


def test_full_restitution_bounces_back_high():
    world, mat = ground_world(restitution=1.0)
    h0 = 3.0
    body = drop_sphere(world, mat, height=h0)
    bounced = False
    rebound_peak = 0.0
    for _ in range(400):
        world.step(DT)
        if world.position[body][1] < 0.6:
            bounced = True
        elif bounced:                               # rising after first impact
            rebound_peak = max(rebound_peak, world.position[body][1])
    assert bounced, 'ball never reached the ground'
    assert rebound_peak > 0.75 * h0                 # e=1 keeps most of its energy


def test_resting_stack_is_stable():
    world, mat = ground_world(restitution=0.0)
    shape = world.add_shape(model.Shape.box((1, 1, 1)))
    bodies = []
    for k in range(5):
        bodies.append(world.add_body(
            model.Motion(type=model.DYNAMIC, mass=1.0),
            collider=model.Collider(shape=shape, physicsMaterial=mat),
            position=(0, 0.5 + k * 1.001, 0)))
    for _ in range(600):
        world.step(DT)
    ys = np.array([world.position[b][1] for b in bodies])
    order = np.argsort(ys)
    assert list(order) == [0, 1, 2, 3, 4]           # stack kept its order
    for k, b in enumerate(bodies):                  # each near its resting slot
        assert world.position[b][1] == pytest.approx(0.5 + k, abs=0.2)
        assert abs(world.position[b][0]) < 0.5      # didn't drift sideways
        assert abs(world.position[b][2]) < 0.5


@pytest.mark.parametrize('angle_deg,friction,should_slide', [
    (40, 0.5, True),      # tan40=0.84 > 0.5
    (10, 0.5, False),     # tan10=0.18 < 0.5
])
def test_incline_slides_iff_tan_exceeds_friction(angle_deg, friction, should_slide):
    theta = np.radians(angle_deg)
    gdir = (np.sin(theta), -np.cos(theta), 0.0)
    world, mat = ground_world(gravity_dir=gdir, friction=friction)
    shape = world.add_shape(model.Shape.box((1, 1, 1)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape, physicsMaterial=mat),
                          position=(0, 0.5, 0))
    for _ in range(400):
        world.step(DT)
    slid = world.position[body][0]
    if should_slide:
        assert slid > 0.5
    else:
        assert slid < 0.05


def test_determinism_bit_identical_runs():
    def run():
        world, mat = ground_world(restitution=0.3)
        shape = world.add_shape(model.Shape.box((1, 1, 1)))
        for k in range(4):
            world.add_body(model.Motion(type=model.DYNAMIC),
                           collider=model.Collider(shape=shape, physicsMaterial=mat),
                           position=(0.01 * k, 1.0 + k * 1.1, 0))
        for _ in range(300):
            world.step(DT)
        return world.position.copy()
    a = run()
    b = run()
    assert np.array_equal(a, b)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
