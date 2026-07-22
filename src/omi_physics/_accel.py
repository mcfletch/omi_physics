"""Whether the compiled accelerators should be used.

Set ``OMI_PHYSICS_NO_ACCEL=1`` to force the pure-Python/NumPy fallback even when
the compiled ``_solver_native`` / ``_collide_native`` modules are importable.
This lets the test matrix exercise both paths without deleting the installed
``.so`` files, and lets a user rule the accelerators in or out when comparing
results. The two paths are meant to agree; see ``docs/ACCELERATORS.md``.
"""
import os


def accelerators_disabled() -> bool:
    """True when ``OMI_PHYSICS_NO_ACCEL`` asks for the pure-Python path."""
    return os.environ.get("OMI_PHYSICS_NO_ACCEL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
