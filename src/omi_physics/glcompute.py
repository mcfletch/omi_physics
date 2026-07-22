"""GPGPU compute backend — the per-body integration kernels on the GPU.

This is the concrete ``GLComputeBackend`` the Phase-7 plan reserved a seam for
(see ``docs/PIPELINE.md`` §Scaling).  It runs the two embarrassingly
parallel stages of ``world.step`` — force integration and position/orientation
integration — as GL 4.3 compute shaders over the world's structure-of-arrays
state, with :class:`~omi_physics.backend.NumpyBackend` as the fallback
whenever a compute context is unavailable.

Scope, stated plainly.  Only the per-body integration kernels run on the GPU; the
broad phase, narrow phase, and sequential-impulse solver stay on the CPU (the
plan calls the solver "the one stage that resists naïve parallelism", deferring
LBVH / graph-colored solving to a future full-residency loop).  Because those
CPU stages read and write the world's numpy arrays, **numpy remains the source of
truth**: each kernel uploads its inputs, dispatches, and reads its outputs back.
The one round-trip this avoids is the intermediate velocity: when a step has no
collision and no joints, force- and position-integration fuse into a single
dispatch (``_fused``), so velocity never leaves the GPU between them.

Numbers: at 100k free bodies the two numpy integrate stages cost ~35 ms/step; the
fused GPU kernel plus transfers is a few ms — a real win *at scale*.  Below a few
thousand awake bodies transfer overhead means numpy is faster, so the default
``auto`` policy in :class:`~omi_physics.world.PhysicsWorld` runs on numpy
and only hands off to this backend once the awake-body count crosses a threshold
(``gpu_threshold``, default 10k).  Force a choice with the world's ``backend`` /
``OPENGLCONTEXT_PHYSICS_BACKEND`` (``numpy``/``gpu``/``auto``).

float32 vs float64.  The world is float64; the GPU computes in float32 (FP64 is
1/32 rate on consumer GPUs and pointless for a game).  Trajectories therefore
match the CPU backend only within tolerance, not bit-for-bit — exactly the
best-effort-on-GPU caveat the plan records for the determinism guarantee.
"""
from typing import Any, List, Tuple, TYPE_CHECKING
import numpy as np

from .backend import NumpyBackend

if TYPE_CHECKING:
    from .world import PhysicsWorld


_FORCES_SRC = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer LinVel { vec4 linvel[]; };
layout(std430, binding = 1) buffer AngVel { vec4 angvel[]; };
layout(std430, binding = 2) buffer Grav   { vec4 grav[]; };
layout(std430, binding = 3) buffer Scal   { vec4 scal[]; };   // gf, linDamp, angDamp, quadDrag
layout(std430, binding = 4) buffer Active { int  actv[]; };
uniform float dt;
uniform float defLin;
uniform float defAng;
uniform uint  nbody;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= nbody || actv[i] == 0) return;
    vec3 v = linvel[i].xyz;
    vec3 w = angvel[i].xyz;
    float gf   = scal[i].x;
    float linD = scal[i].y + defLin;
    float angD = scal[i].z + defAng;
    float qd   = scal[i].w;
    v += grav[i].xyz * gf * dt;
    v *= max(1.0 - linD * dt, 0.0);
    v -= qd * length(v) * v * dt;
    w *= max(1.0 - angD * dt, 0.0);
    linvel[i].xyz = v;
    angvel[i].xyz = w;
}
"""


# Semi-implicit position/orientation update.  The quaternion derivative matches
# mathutil.quat_integrate exactly in (x, y, z, w) order.
_POS_BODY = """
    pos[i].xyz += linvel[i].xyz * dt;
    vec4 q = ori[i];
    vec3 o = angvel[i].xyz;
    vec4 dq = vec4(
        o.x * q.w + o.y * q.z - o.z * q.y,
       -o.x * q.z + o.y * q.w + o.z * q.x,
        o.x * q.y - o.y * q.x + o.z * q.w,
       -o.x * q.x - o.y * q.y - o.z * q.z);
    q += 0.5 * dq * dt;
    ori[i] = normalize(q);
"""

_POS_SRC = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Pos    { vec4 pos[]; };
layout(std430, binding = 1) buffer Ori    { vec4 ori[]; };
layout(std430, binding = 2) buffer LinVel { vec4 linvel[]; };
layout(std430, binding = 3) buffer AngVel { vec4 angvel[]; };
layout(std430, binding = 4) buffer Moving { int  movg[]; };
uniform float dt;
uniform uint  nbody;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= nbody || movg[i] == 0) return;
%s}
""" % _POS_BODY

# Fused kernel: integrate forces (if active) then positions (if moving) in one
# dispatch, so the intermediate velocity never round-trips to the CPU.
_FUSED_SRC = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Pos    { vec4 pos[]; };
layout(std430, binding = 1) buffer Ori    { vec4 ori[]; };
layout(std430, binding = 2) buffer LinVel { vec4 linvel[]; };
layout(std430, binding = 3) buffer AngVel { vec4 angvel[]; };
layout(std430, binding = 4) buffer Grav   { vec4 grav[]; };
layout(std430, binding = 5) buffer Scal   { vec4 scal[]; };
layout(std430, binding = 6) buffer Active { int  actv[]; };
layout(std430, binding = 7) buffer Moving { int  movg[]; };
uniform float dt;
uniform float defLin;
uniform float defAng;
uniform uint  nbody;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= nbody) return;
    if (actv[i] != 0) {
        vec3 v = linvel[i].xyz;
        vec3 w = angvel[i].xyz;
        float gf   = scal[i].x;
        float linD = scal[i].y + defLin;
        float angD = scal[i].z + defAng;
        float qd   = scal[i].w;
        v += grav[i].xyz * gf * dt;
        v *= max(1.0 - linD * dt, 0.0);
        v -= qd * length(v) * v * dt;
        w *= max(1.0 - angD * dt, 0.0);
        linvel[i].xyz = v;
        angvel[i].xyz = w;
    }
    if (movg[i] != 0) {
%s    }
}
""" % _POS_BODY


class _Buffer:
    """A shader-storage buffer sized to the world's body capacity."""

    def __init__(self, gl: Any, ncomp: int) -> None:
        self._gl = gl
        self.ncomp = ncomp                 # 4 for vecN buffers, 1 for flag arrays
        self.id = gl.glGenBuffers(1)
        self.cap = 0

    def ensure(self, n: int) -> None:
        """Grow the buffer to hold at least ``n`` bodies (geometric growth)."""
        if n <= self.cap:
            return
        gl = self._gl
        self.cap = max(8, n, self.cap * 2)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.id)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                        self.cap * self.ncomp * 4, None, gl.GL_DYNAMIC_DRAW)

    def upload(self, data: np.ndarray) -> None:
        """Copy ``data`` into the buffer from element 0.

        ``data`` must be a contiguous 1-D array."""
        # ``data`` must be a contiguous 1-D array (the accelerator mis-sizes 2-D
        # slice views on readback, so keep every transfer flat and contiguous).
        gl = self._gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.id)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)

    def download(self, out: np.ndarray) -> None:
        """Read the buffer back into ``out`` (the bytes land in ``out``'s memory)."""
        # Read into a uint8 view so the accelerator's byte-count check matches the
        # array (it otherwise reads a float array's element count as its byte size
        # and rejects the write); the bytes land in ``out``'s own memory.
        gl = self._gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.id)
        raw = out.view('uint8')
        gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, raw.nbytes, raw)

    def bind(self, binding: int) -> None:
        """Bind the buffer to shader-storage binding point ``binding``."""
        gl = self._gl
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, self.id)


class GLComputeBackend(NumpyBackend):
    """Force/position integration on the GPU; everything else inherited from CPU.

    Requires a current GL 4.3 core context at construction (the compute programs
    compile then).  ``refit_aabbs`` and all collision stages fall through to
    :class:`NumpyBackend`.

    This current implementation is essentially useless because long
    before we reach the scale required to make it worthwhile (10,000+ objects)
    we've stopped being able to process the CPU side updates. In order
    to make a physics engine GPGPU capable we would need to move
    *everything* to the GPU, including the solvers. That would require
    a different algorithm to allow the GPU to do parallel approximation
    vs. iterative refinement.
    """

    name = 'glcompute'

    def __init__(self) -> None:
        from . import backend
        if not backend._has_gl_compute():
            raise RuntimeError('GL 4.3 compute shaders unavailable; use NumpyBackend')
        from OpenGL import GL
        self._gl = GL
        try:
            self._prog_forces = self._compile(_FORCES_SRC)
            self._prog_pos = self._compile(_POS_SRC)
            self._prog_fused = self._compile(_FUSED_SRC)
        except Exception as err:
            raise RuntimeError('compute shader compilation failed: %s' % err)

        self._pos = _Buffer(GL, 4)
        self._ori = _Buffer(GL, 4)
        self._linvel = _Buffer(GL, 4)
        self._angvel = _Buffer(GL, 4)
        self._grav = _Buffer(GL, 4)
        self._scal = _Buffer(GL, 4)
        self._active = _Buffer(GL, 1)
        self._moving = _Buffer(GL, 1)
        self._cap = 0
        # Flat contiguous transfer scratch; ``_f4v`` is a (cap,4) view onto it for
        # packing, uploaded/downloaded as the 1-D ``_f4[:n*4]``.
        self._f4 = np.zeros(0, dtype='f4')
        self._f4v = self._f4.reshape(0, 4)
        self._fi = np.zeros(0, dtype='i4')          # reusable flag staging
        self._fused_done = False

    # -- program helpers -------------------------------------------------
    def _compile(self, src: str) -> int:
        """Compile and link a compute shader; return the GL program id."""
        gl = self._gl
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(shader)
            raise RuntimeError(log.decode() if isinstance(log, bytes) else str(log))
        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program)
            raise RuntimeError(log.decode() if isinstance(log, bytes) else str(log))
        gl.glDeleteShader(shader)
        return program

    def _ensure(self, n: int) -> None:
        """Grow every buffer and the transfer scratch to hold ``n`` bodies."""
        for buf in (self._pos, self._ori, self._linvel, self._angvel,
                    self._grav, self._scal, self._active, self._moving):
            buf.ensure(n)
        if n > self._cap:
            self._cap = max(8, n, self._cap * 2)
            self._f4 = np.zeros(self._cap * 4, dtype='f4')
            self._f4v = self._f4.reshape(self._cap, 4)
            self._fi = np.zeros(self._cap, dtype='i4')

    # -- packing ---------------------------------------------------------
    def _up_vec3(self, buf: _Buffer, arr: np.ndarray, n: int) -> None:
        """Upload the first ``n`` rows of a 3-vector array as padded vec4s."""
        self._f4v[:n, :3] = arr
        self._f4v[:n, 3] = 0.0
        buf.upload(self._f4[:n * 4])

    def _up_vec4(self, buf: _Buffer, arr: np.ndarray, n: int) -> None:
        """Upload the first ``n`` rows of a 4-vector array."""
        self._f4v[:n] = arr
        buf.upload(self._f4[:n * 4])

    def _up_flags(self, buf: _Buffer, mask: np.ndarray, n: int) -> None:
        """Upload the first ``n`` entries of an int flag array."""
        self._fi[:n] = mask
        buf.upload(self._fi[:n])

    def _down_vec3(self, buf: _Buffer, arr: np.ndarray, n: int) -> None:
        """Read the buffer back into the first ``n`` rows of a 3-vector array."""
        buf.download(self._f4[:n * 4])
        arr[:] = self._f4v[:n, :3]

    def _down_vec4(self, buf: _Buffer, arr: np.ndarray, n: int) -> None:
        """Read the buffer back into the first ``n`` rows of a 4-vector array."""
        buf.download(self._f4[:n * 4])
        arr[:] = self._f4v[:n]

    def _scalars(self, world: "PhysicsWorld", n: int) -> None:
        """Pack and upload the per-body scalar tuple (gravity factor, damping, drag)."""
        self._f4v[:n, 0] = world.gravity_factor
        self._f4v[:n, 1] = world.linear_damping
        self._f4v[:n, 2] = world.angular_damping
        self._f4v[:n, 3] = world.quadratic_drag
        self._scal.upload(self._f4[:n * 4])

    def _dispatch(self, program: int, n: int,
                  uniforms: List[Tuple[str, str, Any]]) -> None:
        """Set uniforms, dispatch ``program`` over ``n`` bodies, and barrier on storage.

        Each ``uniforms`` entry is ``(name, kind, value)`` with ``kind`` ``'f'`` for
        a float or anything else for an unsigned int."""
        gl = self._gl
        gl.glUseProgram(program)
        for name, kind, value in uniforms:
            loc = gl.glGetUniformLocation(program, name)
            if kind == 'f':
                gl.glUniform1f(loc, value)
            else:
                gl.glUniform1ui(loc, value)
        groups = (n + 63) // 64
        gl.glDispatchCompute(groups, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT
                           | gl.GL_BUFFER_UPDATE_BARRIER_BIT)

    # -- kernels ---------------------------------------------------------
    def integrate_forces(self, world: "PhysicsWorld", dt: float) -> None:
        """Integrate gravity/damping into the velocities on the GPU.

        With no collision or joints this fuses the position update in too, so
        :meth:`integrate_positions` then skips its own dispatch."""
        n = world.body_count
        if n == 0:
            self._fused_done = False
            return
        self._ensure(n)
        fused = world._collision is None and not world.joint_constraints

        self._up_vec3(self._grav, world.resolve_gravity(), n)
        self._scalars(world, n)
        self._up_flags(self._active, world.dynamic_mask(), n)
        self._up_vec3(self._linvel, world.linear_velocity, n)
        self._up_vec3(self._angvel, world.angular_velocity, n)

        if fused:
            self._up_vec3(self._pos, world.position, n)
            self._up_vec4(self._ori, world.orientation, n)
            self._up_flags(self._moving, world.moving_mask(), n)
            for i, buf in enumerate((self._pos, self._ori, self._linvel,
                                     self._angvel, self._grav, self._scal,
                                     self._active, self._moving)):
                buf.bind(i)
            self._dispatch(self._prog_fused, n, [
                ('dt', 'f', float(dt)),
                ('defLin', 'f', float(world.default_linear_damping)),
                ('defAng', 'f', float(world.default_angular_damping)),
                ('nbody', 'u', n)])
            self._down_vec3(self._pos, world.position, n)
            self._down_vec4(self._ori, world.orientation, n)
            self._down_vec3(self._linvel, world.linear_velocity, n)
            self._down_vec3(self._angvel, world.angular_velocity, n)
            self._fused_done = True
            return

        for binding, buf in enumerate((self._linvel, self._angvel, self._grav,
                                       self._scal, self._active)):
            buf.bind(binding)
        self._dispatch(self._prog_forces, n, [
            ('dt', 'f', float(dt)),
            ('defLin', 'f', float(world.default_linear_damping)),
            ('defAng', 'f', float(world.default_angular_damping)),
            ('nbody', 'u', n)])
        self._down_vec3(self._linvel, world.linear_velocity, n)
        self._down_vec3(self._angvel, world.angular_velocity, n)
        self._fused_done = False

    def integrate_positions(self, world: "PhysicsWorld", dt: float) -> None:
        """Integrate positions and orientations from the velocities on the GPU.

        A no-op when the fused force+position kernel already ran this step."""
        if self._fused_done:
            self._fused_done = False
            return
        n = world.body_count
        if n == 0:
            return
        self._ensure(n)
        self._up_vec3(self._pos, world.position, n)
        self._up_vec4(self._ori, world.orientation, n)
        self._up_vec3(self._linvel, world.linear_velocity, n)
        self._up_vec3(self._angvel, world.angular_velocity, n)
        self._up_flags(self._moving, world.moving_mask(), n)
        for binding, buf in enumerate((self._pos, self._ori, self._linvel,
                                       self._angvel, self._moving)):
            buf.bind(binding)
        self._dispatch(self._prog_pos, n, [
            ('dt', 'f', float(dt)), ('nbody', 'u', n)])
        self._down_vec3(self._pos, world.position, n)
        self._down_vec4(self._ori, world.orientation, n)
