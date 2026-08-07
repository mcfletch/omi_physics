"""omi_physics — a real-time rigid-body engine built on the OMI glTF physics model.

The package is a self-contained, renderer-agnostic physics engine. State lives in
a flat structure-of-arrays (:mod:`omi_physics.world`), the data model is the OMI
glTF physics schema (:mod:`omi_physics.model`), and the whole step pipeline —
broadphase, narrowphase, sequential-impulse solver, integration — is pure NumPy
with optional compiled accelerators (:mod:`omi_physics._solver_native`,
:mod:`omi_physics._collide_native`) that drop in transparently when present.

Only NumPy is required. PyOpenGL is an *optional* dependency used solely by the
GPU compute backend (:mod:`omi_physics.glcompute`); import and use of the engine
never touches GL unless you explicitly select the GPU backend.

.. warning::

   This code is **largely LLM-written**. It has a test suite (see ``tests/``) and
   the CPU backend is deterministic run-to-run, but it comes with **no
   guarantees** of correctness, accuracy, or fitness for any purpose (see the
   MIT ``LICENSE``). Review it before relying on it for anything that matters.
"""
from . import model
from . import mathutil
from .world import PhysicsWorld
from .backend import NumpyBackend, select_backend

__version__ = "0.3.0"

__all__ = [
    "PhysicsWorld",
    "NumpyBackend",
    "select_backend",
    "model",
    "mathutil",
    "__version__",
]
