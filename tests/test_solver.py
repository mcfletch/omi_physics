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


def test_a_crushed_body_does_not_go_through_the_floor():
    """A tenth of a gramme under 243 kilogrammes stays on top of the ground.

    The stack itself does not survive these ratios and is not asked to:
    sequential impulses pass a load down one contact at a time, so the light
    box is squeezed out sideways. What it may not do is be squeezed *through*
    the floor. It is held in place by two corrections that cancel -- the floor
    pushing it up and the box above pushing it down, both moving only the light
    body because only it has any inverse mass to speak of -- and when the one
    against the floor is the one that loses, the box sinks a little each step.
    Once it passes the floor's mid-plane the separating axis is nearer the far
    side, the contact normal points down, and the floor throws it out
    underneath: it fell to y = -43 before this was fixed.

    Ratios like these are ordinary: a crate of ammunition under a truck, a
    dropped screw under a door.
    """
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), sleep_enabled=False)
    floor = world.add_shape(model.Shape.box((400, 1, 400)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=floor), position=(0, -0.5, 0))
    cube = world.add_shape(model.Shape.box((1, 1, 1)))
    stack = [world.add_body(model.Motion(type=model.DYNAMIC, mass=mass),
                            collider=model.Collider(shape=cube),
                            position=(0, 0.5 + k * 1.05, 0))
             for k, mass in enumerate((1e-4, 12.0, 231.0))]
    lowest = np.inf
    for _ in range(400):
        world.step(DT)
        lowest = min(lowest, world.position[stack][:, 1].min())
    # The floor's top is y=0 and a box is 1 deep, so a box resting on it sits at
    # 0.5. Anything at or below -0.5 has its centre at the floor's own centre,
    # which is the point of no return.
    assert lowest > -0.5, 'a box was pushed into the floor as far as its middle'
    resting = world.position[stack]
    still_over_the_floor = np.all(np.abs(resting[:, [0, 2]]) < 190.0, axis=1)
    assert np.all(resting[still_over_the_floor][:, 1] > -0.5)


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


@pytest.mark.parametrize('heavy', [10.0, 75.0, 1000.0])
def test_a_heavy_body_does_not_push_a_light_one_through_the_floor(heavy):
    """The position pass corrects each contact against what it has already done.

    A box meets a floor at four points that all report the same overlap. Four
    full corrections push it four times as far as it is in, so the pair comes
    back the next step further out of place than it started -- and with a heavy
    body pressing a light one down, that grows instead of settling until the
    light one is thrown out the far side of the floor.
    """
    world, mat = ground_world(restitution=0.0)
    cube = world.add_shape(model.Shape.box((1, 1, 1)))
    light = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                           collider=model.Collider(shape=cube, physicsMaterial=mat),
                           position=(0, 0.5, 0))
    world.add_body(model.Motion(type=model.DYNAMIC, mass=heavy),
                   collider=model.Collider(shape=cube, physicsMaterial=mat),
                   position=(0, 1.55, 0))
    for _ in range(400):
        world.step(DT)
    assert world.position[light][1] == pytest.approx(0.5, abs=0.1)


def test_a_body_placed_deep_inside_another_climbs_out():
    """Depenetration is bounded per step, so recovery is a handful of frames
    rather than a body flung across the level in one."""
    world, mat = ground_world(restitution=0.0)
    shape = world.add_shape(model.Shape.sphere(0.5))
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape, physicsMaterial=mat),
                          position=(0, -0.4, 0))          # most of the way in
    for _ in range(240):
        world.step(DT)
    assert world.position[body][1] == pytest.approx(0.5, abs=0.05)


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
