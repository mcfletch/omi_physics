"""OMI glTF physics import/export.

Reads the ``OMI_physics_shape`` / ``OMI_physics_body`` / ``OMI_physics_gravity`` /
``OMI_physics_joint`` extension blocks of a glTF document straight into
:mod:`omi_physics.model` structures, and writes them back out.  Because
the model *is* the OMI schema, load→export is a near-identity round-trip.  The
``KHR_physics_rigid_bodies`` / glTF-2.1 ``shapes`` readers drop onto the same
structures when they ratify.

This module works on plain glTF JSON dicts (no GL, no mesh loading required), so
the whole path is unit-testable.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from . import model

SUPPORTED = ('OMI_physics_shape', 'OMI_physics_body',
             'OMI_physics_gravity', 'OMI_physics_joint')


@dataclass
class NodeBody:
    """The OMI physics attached to a single glTF node (any field may be absent)."""
    motion: Optional[model.Motion] = None
    collider: Optional[model.Collider] = None
    trigger: Optional[model.Trigger] = None
    gravity: Optional[model.Gravity] = None
    joint: Optional[model.JointAttach] = None


@dataclass
class PhysicsDocument:
    """The document-level OMI physics tables plus per-node bodies keyed by node index."""
    shapes: List[model.Shape] = field(default_factory=list)
    materials: List[model.Material] = field(default_factory=list)
    filters: List[model.CollisionFilter] = field(default_factory=list)
    joints: List[model.Joint] = field(default_factory=list)
    global_gravity: Optional[model.Gravity] = None
    node_bodies: Dict[int, NodeBody] = field(default_factory=dict)


# ── shapes ──────────────────────────────────────────────────────────────
def _read_shape(entry: dict) -> model.Shape:
    """Build a :class:`model.Shape` from one ``OMI_physics_shape`` entry."""
    t = entry['type']
    p = entry.get(t, {})
    if t == 'box':
        return model.Shape.box(p.get('size', [1, 1, 1]))
    if t == 'sphere':
        return model.Shape.sphere(p.get('radius', 0.5))
    if t == 'capsule':
        return model.Shape.capsule(p.get('height', 1.0), p.get('radius', 0.5))
    if t == 'cylinder':
        return model.Shape(type='cylinder', height=p.get('height', 2.0),
                           radiusBottom=p.get('radiusBottom', 0.5),
                           radiusTop=p.get('radiusTop', 0.5))
    if t in ('convex', 'trimesh'):
        return model.Shape(type=t, mesh=p.get('mesh', -1))
    raise ValueError('unknown OMI shape type %r' % t)


def _write_shape(shape: model.Shape) -> dict:
    """Serialize a :class:`model.Shape` to its ``OMI_physics_shape`` entry."""
    t = shape.type
    if t == 'box':
        return {'type': 'box', 'box': {'size': list(shape.size)}}
    if t == 'sphere':
        return {'type': 'sphere', 'sphere': {'radius': shape.radius}}
    if t == 'capsule':
        return {'type': 'capsule',
                'capsule': {'height': shape.height, 'radius': shape.radiusBottom}}
    if t == 'cylinder':
        return {'type': 'cylinder',
                'cylinder': {'height': shape.height,
                             'radiusBottom': shape.radiusBottom,
                             'radiusTop': shape.radiusTop}}
    return {'type': t, t: {'mesh': shape.mesh}}


# ── materials / filters ─────────────────────────────────────────────────
def _read_material(d: dict) -> model.Material:
    """Build a :class:`model.Material` from a physics-material entry."""
    return model.Material(
        staticFriction=d.get('staticFriction', 0.6),
        dynamicFriction=d.get('dynamicFriction', 0.6),
        restitution=d.get('restitution', 0.0),
        frictionCombine=d.get('frictionCombine', model.AVERAGE),
        restitutionCombine=d.get('restitutionCombine', model.AVERAGE))


def _write_material(m: model.Material) -> dict:
    """Serialize a :class:`model.Material` to its physics-material entry."""
    return {'staticFriction': m.staticFriction, 'dynamicFriction': m.dynamicFriction,
            'restitution': m.restitution, 'frictionCombine': m.frictionCombine,
            'restitutionCombine': m.restitutionCombine}


def _read_filter(d: dict) -> model.CollisionFilter:
    """Build a :class:`model.CollisionFilter` from a collision-filter entry."""
    return model.CollisionFilter(
        collisionSystems=tuple(d.get('collisionSystems', [])),
        collideWithSystems=(tuple(d['collideWithSystems'])
                            if 'collideWithSystems' in d else None),
        notCollideWithSystems=tuple(d.get('notCollideWithSystems', [])))


def _write_filter(f: model.CollisionFilter) -> dict:
    """Serialize a :class:`model.CollisionFilter` to its collision-filter entry."""
    out = {'collisionSystems': list(f.collisionSystems),
           'notCollideWithSystems': list(f.notCollideWithSystems)}
    if f.collideWithSystems is not None:
        out['collideWithSystems'] = list(f.collideWithSystems)
    return out


# ── motion / collider / trigger ─────────────────────────────────────────
def _read_motion(d: dict) -> model.Motion:
    """Build a :class:`model.Motion` from a body ``motion`` block."""
    return model.Motion(
        type=d['type'],
        mass=d.get('mass', 1.0),
        centerOfMass=tuple(d.get('centerOfMass', (0, 0, 0))),
        inertiaDiagonal=tuple(d.get('inertiaDiagonal', (0, 0, 0))),
        inertiaOrientation=tuple(d.get('inertiaOrientation', (0, 0, 0, 1))),
        linearVelocity=tuple(d.get('linearVelocity', (0, 0, 0))),
        angularVelocity=tuple(d.get('angularVelocity', (0, 0, 0))),
        gravityFactor=d.get('gravityFactor', 1.0))


def _write_motion(m: model.Motion) -> dict:
    """Serialize a :class:`model.Motion` to a body ``motion`` block."""
    return {'type': m.type, 'mass': m.mass, 'centerOfMass': list(m.centerOfMass),
            'inertiaDiagonal': list(m.inertiaDiagonal),
            'inertiaOrientation': list(m.inertiaOrientation),
            'linearVelocity': list(m.linearVelocity),
            'angularVelocity': list(m.angularVelocity),
            'gravityFactor': m.gravityFactor}


def _read_collider(d: dict) -> model.Collider:
    """Build a :class:`model.Collider` from a body ``collider`` block."""
    return model.Collider(shape=d.get('shape', -1),
                          physicsMaterial=d.get('physicsMaterial', -1),
                          collisionFilter=d.get('collisionFilter', -1))


def _write_collider(c: model.Collider) -> dict:
    """Serialize a :class:`model.Collider` to a body ``collider`` block."""
    out = {'shape': c.shape}
    if c.physicsMaterial >= 0:
        out['physicsMaterial'] = c.physicsMaterial
    if c.collisionFilter >= 0:
        out['collisionFilter'] = c.collisionFilter
    return out


# ── gravity ─────────────────────────────────────────────────────────────
def _read_gravity(d: dict) -> model.Gravity:
    """Build a :class:`model.Gravity` from a gravity block or volume entry."""
    return model.Gravity(
        type=d.get('type', model.DIRECTIONAL),
        gravity=d.get('gravity', 9.81),
        direction=tuple(d.get('direction', (0, -1, 0))),
        priority=d.get('priority', 0),
        replace=d.get('replace', False),
        stop=d.get('stop', False),
        center=tuple(d.get('center', (0, 0, 0))))


def _write_global_gravity(gv: model.Gravity) -> dict:
    """Serialize a scene-wide :class:`model.Gravity` to the top-level gravity block."""
    return {'gravity': gv.gravity, 'direction': list(gv.direction)}


def _write_gravity_volume(gv: model.Gravity) -> dict:
    """Serialize a per-node :class:`model.Gravity` volume to its node extension block."""
    out = {'type': gv.type, 'gravity': gv.gravity, 'priority': gv.priority,
           'replace': gv.replace, 'stop': gv.stop}
    if gv.type == model.POINT:
        out['center'] = list(gv.center)
    else:
        out['direction'] = list(gv.direction)
    return out


# ── joints ──────────────────────────────────────────────────────────────
def _read_joint(d: dict) -> model.Joint:
    """Build a :class:`model.Joint` (its limits and drives) from a joint entry."""
    limits = [model.JointLimit(
        linearAxes=tuple(l.get('linearAxes', [])),
        angularAxes=tuple(l.get('angularAxes', [])),
        min=l.get('min', -np.inf), max=l.get('max', np.inf),
        stiffness=l.get('stiffness', np.inf), damping=l.get('damping', 0.0))
        for l in d.get('limits', [])]
    drives = [model.JointDrive(
        type=dr.get('type', 'linear'), mode=dr.get('mode', 'force'),
        axis=dr.get('axis', 0), maxForce=dr.get('maxForce', np.inf),
        positionTarget=dr.get('positionTarget'), velocityTarget=dr.get('velocityTarget'),
        stiffness=dr.get('stiffness', 0.0), damping=dr.get('damping', 0.0))
        for dr in d.get('drives', [])]
    return model.Joint(limits=limits, drives=drives)


def _read_joint_attach(d: dict) -> model.JointAttach:
    """Build a :class:`model.JointAttach` from a body ``joint`` block."""
    return model.JointAttach(joint=d.get('joint', -1),
                             connectedNode=d.get('connectedNode', -1),
                             enableCollision=d.get('enableCollision', False))


def _write_joint_attach(j: model.JointAttach) -> dict:
    """Serialize a :class:`model.JointAttach` to a body ``joint`` block."""
    return {'joint': j.joint, 'connectedNode': j.connectedNode,
            'enableCollision': j.enableCollision}


# ── document load / export ──────────────────────────────────────────────
def load_document(gltf: dict) -> PhysicsDocument:
    """Read all OMI physics extension blocks of a glTF JSON dict into a document."""
    doc = PhysicsDocument()
    top = gltf.get('extensions', {}) or {}
    shape_ext = top.get('OMI_physics_shape', {})
    for entry in shape_ext.get('shapes', []):
        doc.shapes.append(_read_shape(entry))
    body_ext = top.get('OMI_physics_body', {})
    for m in body_ext.get('physicsMaterials', []):
        doc.materials.append(_read_material(m))
    for f in body_ext.get('collisionFilters', []):
        doc.filters.append(_read_filter(f))
    joint_ext = top.get('OMI_physics_joint', {})
    for j in joint_ext.get('physicsJoints', []):
        doc.joints.append(_read_joint(j))
    grav_ext = top.get('OMI_physics_gravity')
    if grav_ext:
        doc.global_gravity = _read_gravity(grav_ext)

    for idx, node in enumerate(gltf.get('nodes', [])):
        ext = node.get('extensions', {}) or {}
        nb = NodeBody()
        found = False
        body = ext.get('OMI_physics_body')
        if body:
            found = True
            if 'motion' in body:
                nb.motion = _read_motion(body['motion'])
            if 'collider' in body:
                nb.collider = _read_collider(body['collider'])
            if 'trigger' in body:
                nb.trigger = model.Trigger(shape=body['trigger'].get('shape', -1))
            if 'joint' in body:
                nb.joint = _read_joint_attach(body['joint'])
        grav = ext.get('OMI_physics_gravity')
        if grav:
            found = True
            nb.gravity = _read_gravity(grav)
        if found:
            doc.node_bodies[idx] = nb
    return doc


def export_extensions(doc: PhysicsDocument) -> Tuple[dict, dict]:
    """Return ``(top_extensions, node_extensions)`` reproducing the OMI blocks.

    ``node_extensions`` maps node index to that node's ``extensions`` dict.
    """
    top: Dict[str, dict] = {}
    if doc.shapes:
        top['OMI_physics_shape'] = {'shapes': [_write_shape(s) for s in doc.shapes]}
    body: Dict[str, list] = {}
    if doc.materials:
        body['physicsMaterials'] = [_write_material(m) for m in doc.materials]
    if doc.filters:
        body['collisionFilters'] = [_write_filter(f) for f in doc.filters]
    if body:
        top['OMI_physics_body'] = body
    if doc.joints:
        top['OMI_physics_joint'] = {
            'physicsJoints': [_write_joint(j) for j in doc.joints]}
    if doc.global_gravity is not None:
        top['OMI_physics_gravity'] = _write_global_gravity(doc.global_gravity)

    node_ext: Dict[int, dict] = {}
    for idx, nb in doc.node_bodies.items():
        ext: Dict[str, dict] = {}
        node_body: Dict[str, dict] = {}
        if nb.motion is not None:
            node_body['motion'] = _write_motion(nb.motion)
        if nb.collider is not None:
            node_body['collider'] = _write_collider(nb.collider)
        if nb.trigger is not None:
            node_body['trigger'] = {'shape': nb.trigger.shape}
        if nb.joint is not None:
            node_body['joint'] = _write_joint_attach(nb.joint)
        if node_body:
            ext['OMI_physics_body'] = node_body
        if nb.gravity is not None:
            ext['OMI_physics_gravity'] = _write_gravity_volume(nb.gravity)
        node_ext[idx] = ext
    return top, node_ext


def _write_joint(j: model.Joint) -> dict:
    """Serialize a :class:`model.Joint` (its limits and drives) to a joint entry."""
    out: Dict[str, list] = {}
    if j.limits:
        out['limits'] = [_write_limit(l) for l in j.limits]
    if j.drives:
        out['drives'] = [_write_drive(d) for d in j.drives]
    return out


def _write_limit(l: model.JointLimit) -> dict:
    """Serialize a :class:`model.JointLimit`; infinite min/max are omitted."""
    out: Dict[str, object] = {'linearAxes': list(l.linearAxes),
                              'angularAxes': list(l.angularAxes)}
    if np.isfinite(l.min):
        out['min'] = l.min
    if np.isfinite(l.max):
        out['max'] = l.max
    return out


def _write_drive(d: model.JointDrive) -> dict:
    """Serialize a :class:`model.JointDrive`; an infinite ``maxForce`` is omitted."""
    out = {'type': d.type, 'mode': d.mode, 'axis': d.axis}
    if d.positionTarget is not None:
        out['positionTarget'] = d.positionTarget
    if d.velocityTarget is not None:
        out['velocityTarget'] = d.velocityTarget
    if np.isfinite(d.maxForce):
        out['maxForce'] = d.maxForce
    return out
