#!/usr/bin/env python
"""Build the optional compiled physics accelerators.

All project metadata lives in ``pyproject.toml``; this file exists only to build
the two Cython extensions. Like ``OpenGL_accelerate`` / ``vrml_accelerate`` these
are a pure speedup: :mod:`omi_physics` imports them when present and falls back to
identical NumPy/Python code when they are absent, so a source install without a C
compiler still works. Prebuilt wheels (cibuildwheel) ship the binaries so end
users need no compiler.
"""
import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Extension name -> path to its .pyx (falling back to a shipped .c).
_ACCEL_STEMS = [
    "src/omi_physics/_solver_native",
    "src/omi_physics/_collide_native",
]


def accelerator_extensions():
    """Cythonize the accelerator .pyx files, or fall back to shipped .c sources."""
    exts = []
    for stem in _ACCEL_STEMS:
        name = "omi_physics." + os.path.basename(stem)
        pyx, csrc = stem + ".pyx", stem + ".c"
        if os.path.exists(pyx):
            exts.append(Extension(name, [pyx], extra_compile_args=["-O3"]))
        elif os.path.exists(csrc):
            exts.append(Extension(name, [csrc], extra_compile_args=["-O3"]))
    try:
        from Cython.Build import cythonize
        return cythonize(exts, language_level=3, quiet=True)
    except ImportError:
        return [e for e in exts if e.sources[0].endswith(".c")]


class optional_build_ext(_build_ext):
    """Build the accelerators if the toolchain allows; never fail the install."""

    def run(self):
        try:
            _build_ext.run(self)
        except Exception as err:
            sys.stderr.write(
                "WARNING: omi_physics accelerators not built (%s); "
                "using pure-Python fallback\n" % (err,)
            )

    def build_extension(self, ext):
        try:
            _build_ext.build_extension(self, ext)
        except Exception as err:
            sys.stderr.write("WARNING: skipping %s (%s)\n" % (ext.name, err))


if __name__ == "__main__":
    setup(
        ext_modules=accelerator_extensions(),
        cmdclass={"build_ext": optional_build_ext},
    )
