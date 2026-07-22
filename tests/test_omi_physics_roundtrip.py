"""Phase 2 glTF OMI import/export round-trip (no GL)."""
import copy
import pytest

from omi_physics import model
from omi_physics import omi_gltf


def fixture_gltf():
    return {
        'extensionsUsed': list(omi_gltf.SUPPORTED),
        'extensions': {
            'OMI_physics_shape': {'shapes': [
                {'type': 'box', 'box': {'size': [2.0, 1.0, 3.0]}},
                {'type': 'sphere', 'sphere': {'radius': 0.75}},
                {'type': 'capsule', 'capsule': {'height': 1.5, 'radius': 0.4}},
                {'type': 'cylinder', 'cylinder': {'height': 2.0,
                                                  'radiusBottom': 0.5, 'radiusTop': 0.5}},
                {'type': 'convex', 'convex': {'mesh': 0}},
                {'type': 'trimesh', 'trimesh': {'mesh': 1}},
            ]},
            'OMI_physics_body': {
                'physicsMaterials': [
                    {'staticFriction': 0.8, 'dynamicFriction': 0.7, 'restitution': 0.3,
                     'frictionCombine': 'average', 'restitutionCombine': 'maximum'}],
                'collisionFilters': [
                    {'collisionSystems': ['props'], 'notCollideWithSystems': ['debris']}],
            },
            'OMI_physics_joint': {'physicsJoints': [
                {'limits': [{'linearAxes': [0, 1, 2], 'min': 0.0, 'max': 0.0}],
                 'drives': [{'type': 'angular', 'mode': 'force', 'axis': 1,
                             'velocityTarget': 3.0, 'maxForce': 50.0}]}]},
            'OMI_physics_gravity': {'gravity': 9.81, 'direction': [0.0, -1.0, 0.0]},
        },
        'nodes': [
            {'name': 'crate', 'translation': [0, 5, 0],
             'extensions': {'OMI_physics_body': {
                 'motion': {'type': 'dynamic', 'mass': 2.0, 'centerOfMass': [0, 0, 0],
                            'inertiaDiagonal': [0, 0, 0], 'inertiaOrientation': [0, 0, 0, 1],
                            'linearVelocity': [0, 0, 0], 'angularVelocity': [0, 0, 0],
                            'gravityFactor': 1.0},
                 'collider': {'shape': 0, 'physicsMaterial': 0, 'collisionFilter': 0}}}},
            {'name': 'ground',
             'extensions': {'OMI_physics_body': {
                 'motion': {'type': 'static', 'mass': 1.0, 'centerOfMass': [0, 0, 0],
                            'inertiaDiagonal': [0, 0, 0], 'inertiaOrientation': [0, 0, 0, 1],
                            'linearVelocity': [0, 0, 0], 'angularVelocity': [0, 0, 0],
                            'gravityFactor': 1.0},
                 'collider': {'shape': 5}}}},
            {'name': 'zone',
             'extensions': {'OMI_physics_body': {'trigger': {'shape': 0}}}},
            {'name': 'planet',
             'extensions': {'OMI_physics_gravity': {
                 'type': 'point', 'gravity': 20.0, 'priority': 3,
                 'replace': True, 'stop': False, 'center': [0, 0, 0]}}},
        ],
    }


def test_load_reads_exact_fields():
    doc = omi_gltf.load_document(fixture_gltf())
    assert [s.type for s in doc.shapes] == \
        ['box', 'sphere', 'capsule', 'cylinder', 'convex', 'trimesh']
    assert tuple(doc.shapes[0].size) == (2.0, 1.0, 3.0)
    assert doc.shapes[1].radius == 0.75
    assert doc.shapes[4].mesh == 0
    assert doc.materials[0].restitution == 0.3
    assert doc.materials[0].restitutionCombine == 'maximum'
    assert doc.filters[0].collisionSystems == ('props',)
    assert doc.global_gravity.gravity == 9.81
    assert doc.joints[0].drives[0].velocityTarget == 3.0

    crate = doc.node_bodies[0]
    assert crate.motion.type == 'dynamic' and crate.motion.mass == 2.0
    assert crate.collider.shape == 0 and crate.collider.physicsMaterial == 0
    assert doc.node_bodies[1].motion.type == 'static'    # immobile Earth
    assert doc.node_bodies[2].trigger.shape == 0
    planet = doc.node_bodies[3].gravity
    assert planet.type == 'point' and planet.replace and planet.gravity == 20.0


def test_export_reproduces_json():
    original = fixture_gltf()
    doc = omi_gltf.load_document(original)
    top, node_ext = omi_gltf.export_extensions(doc)
    assert top['OMI_physics_shape'] == original['extensions']['OMI_physics_shape']
    assert top['OMI_physics_body']['physicsMaterials'] == \
        original['extensions']['OMI_physics_body']['physicsMaterials']
    assert top['OMI_physics_gravity'] == original['extensions']['OMI_physics_gravity']
    for idx, node in enumerate(original['nodes']):
        assert node_ext[idx] == node['extensions']


def test_static_type_means_immobile():
    doc = omi_gltf.load_document(fixture_gltf())
    assert doc.node_bodies[1].motion.type == model.STATIC


def test_reexport_is_stable():
    doc1 = omi_gltf.load_document(fixture_gltf())
    top, node_ext = omi_gltf.export_extensions(doc1)
    rebuilt = fixture_gltf()
    rebuilt['extensions'] = top
    for idx, ext in node_ext.items():
        rebuilt['nodes'][idx]['extensions'] = ext
    doc2 = omi_gltf.load_document(rebuilt)
    top2, node_ext2 = omi_gltf.export_extensions(doc2)
    assert top == top2 and node_ext == node_ext2


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
