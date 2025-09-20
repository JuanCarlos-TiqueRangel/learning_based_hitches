import time
import numpy as np
import mujoco
import mujoco.viewer
import os

import matplotlib.pyplot as plt
from collections import deque


try:
    mujoco.mj_setNumThreads(max(1, os.cpu_count() - 1))
except Exception:
    pass


MODEL = "../model/plugin/elasticity/RLhitches.xml"

# ------------------- load -------------------
model = mujoco.MjModel.from_xml_path(MODEL)
data  = mujoco.MjData(model)

def gname(g):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""



############################### SNAPSHOT
SNAP_PATH = "hitch_snapshot.npz"

def save_snapshot(model, data, path=SNAP_PATH):
    np.savez_compressed(
        path,
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        act=getattr(data, "act", np.empty(0)).copy(),
        ctrl=data.ctrl.copy(),
        qacc_warmstart=getattr(data, "qacc_warmstart", np.empty(0)).copy(),
        plugin_state=getattr(data, "plugin_state", np.empty(0)).copy(),
    )
    print(f"[snapshot] saved → {os.path.abspath(path)}")

def load_snapshot(model, data, path=SNAP_PATH):
    Z = np.load(path, allow_pickle=False)
    def _copy(dst, key):
        if key in Z and dst.size:
            n = min(dst.size, Z[key].size)
            dst.flat[:n] = Z[key].flat[:n]
    _copy(data.qpos, "qpos")
    _copy(data.qvel, "qvel")
    if hasattr(data, "act"): _copy(data.act, "act")
    _copy(data.ctrl, "ctrl")
    if hasattr(data, "qacc_warmstart"): _copy(data.qacc_warmstart, "qacc_warmstart")
    if hasattr(data, "plugin_state"):   _copy(data.plugin_state,   "plugin_state")
    mujoco.mj_forward(model, data)
    data.time = 0.0
    print(f"[snapshot] restored ← {os.path.abspath(path)}")


load_snapshot(model, data, SNAP_PATH)


# collect the capsule geoms that belong to each cable (by name)
RA = [g for g in range(model.ngeom)
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and "RAG" in gname(g)]
RB = [g for g in range(model.ngeom)
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and "RBG" in gname(g)]
print(f"[collect] RA={len(RA)}  RB={len(RB)}", flush=True)
if not RA or not RB:
    raise RuntimeError("Could not find RAG*/RBG* capsule geoms. Check names / XML.")

# allow per-geom colors if the plugin isn't overriding them (material off)
for g in RA + RB:
    model.geom_matid[g] = -1

# precompute index arrays and base radii (so runtime thickness changes don't affect gap math)
RA_idx = np.array(RA, dtype=int)
RB_idx = np.array(RB, dtype=int)
BASE_RADIUS = model.geom_size[:, 0].copy()       # keep originals to restore each frame (visual)
BASE_RGBA   = model.geom_rgba.copy()
RA_r0 = BASE_RADIUS[RA_idx].copy()               # use base radii for gap computation
RB_r0 = BASE_RADIUS[RB_idx].copy()


# # 1) Collision masks
# #   RA: contype=1, conaffinity=2 ; RB: contype=2, conaffinity=1
# model.geom_contype[RA,]     = 1
# model.geom_conaffinity[RA,] = 2
# model.geom_contype[RB,]     = 2
# model.geom_conaffinity[RB,] = 1

# ---- make hitches hold: frictional contact on cable capsules ----
ids = np.array(RA + RB, dtype=int)
model.geom_matid[ids]   = -1          # # ensure per-geom colors can show
model.geom_condim[ids]  = 4           # 6 # enable sliding + torsional + rolling friction
model.geom_friction[ids, 0] = 0.1     # 0.1 # slide
model.geom_friction[ids, 1] = 0.2   # 0.005 # spin
model.geom_friction[ids, 2] = 0.0  # 0.0001 # roll
model.geom_margin[ids]  = 0.0      # 0.0005 # small positive margin helps stability

# Optional global contact softening (stability)
model.opt.o_solimp[:] = np.array([0.9, 1.0, 0.001, 0.5, 2.0])
model.opt.o_solref[:] = np.array([0.02, 1.0])
# #model.opt.timestep    = 0.001  # tighter time step during knotting

model.opt.noslip_iterations = 20
model.opt.noslip_tolerance = 1e-6  # (default tolerance is fine)

# 3) Solver/timestep
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.opt.solver     = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 100
model.opt.tolerance  = 1e-6



plt.ion()
fig, (ax_speed, ax_vec) = plt.subplots(1, 2, figsize=(11, 4.5))
try:
    fig.canvas.manager.set_window_title("Robot speeds (A,B) & commanded velocity vectors")
except Exception:
    pass

# history buffers (last N seconds of data)
HORIZON_S = 100.0
hist_t   = deque(maxlen=2000)
hist_vA  = deque(maxlen=2000)  # actual |v| for A
hist_vB  = deque(maxlen=2000)  # actual |v| for B
hist_vAc = deque(maxlen=2000)  # commanded |v| for A
hist_vBc = deque(maxlen=2000)  # commanded |v| for B

# line objects for magnitudes
ln_vA , = ax_speed.plot([], [], label="A |v| actual")
ln_vB , = ax_speed.plot([], [], label="B |v| actual")
ln_vAc, = ax_speed.plot([], [], linestyle="--", label="A |v| commanded")
ln_vBc, = ax_speed.plot([], [], linestyle="--", label="B |v| commanded")
ax_speed.set_xlabel("time [s]")
ax_speed.set_ylabel("speed [m/s]")
ax_speed.legend(loc="upper right")
ax_speed.grid(True, alpha=0.3)

# vector plot axes (2D XY vectors)
ax_vec.set_title("Commanded velocity vectors in XY (A: left, B: right)")
ax_vec.set_aspect("equal", adjustable="box")
ax_vec.set_xlim(-1.2, 1.2)
ax_vec.set_ylim(-1.2, 1.2)
ax_vec.grid(True, alpha=0.3)

# quiver: initialize ONCE with zero-length arrows for A and B
anchors_x = np.array([-0.5, 0.5], dtype=float)   # A left, B right
anchors_y = np.array([ 0.0, 0.0], dtype=float)
q_cmd = ax_vec.quiver(anchors_x, anchors_y, [0.0, 0.0], [0.0, 0.0],
                      angles='xy', scale_units='xy', scale=1, color=['C0','C1'])

# previous samples for finite-difference
_prev_sample_time = 0.0
_prev_posA = None
_prev_posB = None
_prev_A_cmd = None
_prev_B_cmd = None

# sampling cadence for the plots (sim seconds)
PLOT_DT_SIM = 0.01  # 50 ms of simulation time
_last_plot_sim_time = 0.0

def _update_live_plots(sim_time, A_cmd, B_cmd):
    """
    Update speed plots and commanded velocity vectors.
    Uses one persistent quiver (q_cmd) and updates it with set_UVC/set_offsets.
    """
    global _prev_sample_time, _prev_posA, _prev_posB, _prev_A_cmd, _prev_B_cmd

    # Initialize on first call
    if _prev_posA is None or _prev_A_cmd is None:
        p = get_robot_positions()
        _prev_posA = p["A_point"].copy()
        _prev_posB = p["B_point"].copy()
        _prev_A_cmd = A_cmd.copy()
        _prev_B_cmd = B_cmd.copy()
        _prev_sample_time = sim_time
        return

    dt = max(1e-9, sim_time - _prev_sample_time)

    # Actual velocities from positions
    p = get_robot_positions()
    posA, posB = p["A_point"], p["B_point"]
    vA = (posA - _prev_posA) / dt
    vB = (posB - _prev_posB) / dt

    # Commanded velocities from A_cmd/B_cmd ramp
    vA_cmd = (A_cmd - _prev_A_cmd) / dt
    vB_cmd = (B_cmd - _prev_B_cmd) / dt

    # Update histories
    hist_t.append(sim_time)
    hist_vA.append(float(np.linalg.norm(vA)))
    hist_vB.append(float(np.linalg.norm(vB)))
    hist_vAc.append(float(np.linalg.norm(vA_cmd)))
    hist_vBc.append(float(np.linalg.norm(vB_cmd)))

    # Keep last HORIZON_S seconds
    while len(hist_t) >= 2 and (hist_t[-1] - hist_t[0] > HORIZON_S):
        hist_t.popleft(); hist_vA.popleft(); hist_vB.popleft(); hist_vAc.popleft(); hist_vBc.popleft()

    # ---- magnitude plot
    ln_vA.set_data(hist_t, hist_vA)
    ln_vB.set_data(hist_t, hist_vB)
    ln_vAc.set_data(hist_t, hist_vAc)
    ln_vBc.set_data(hist_t, hist_vBc)
    if hist_t:
        ax_speed.set_xlim(hist_t[0], max(hist_t[-1], hist_t[0] + 2.0))
        ymax = max(1e-6, max(max(hist_vA) if hist_vA else 0,
                             max(hist_vB) if hist_vB else 0,
                             max(hist_vAc) if hist_vAc else 0,
                             max(hist_vBc) if hist_vBc else 0))
        ax_speed.set_ylim(0.0, ymax * 1.25)

    # ---- vector plot (update existing quiver)
    # Update anchors (optional: keep static)
    q_cmd.set_offsets(np.c_[anchors_x, anchors_y])
    # Update U,V to commanded XY velocities
    U = np.array([vA_cmd[0], vB_cmd[0]], dtype=float)
    V = np.array([vA_cmd[1], vB_cmd[1]], dtype=float)
    q_cmd.set_UVC(U, V)
    ax_vec.set_title(f"Commanded velocity vectors XY  |  "
                     f"|vA_cmd|={np.linalg.norm(vA_cmd):.2f} m/s, "
                     f"|vB_cmd|={np.linalg.norm(vB_cmd):.2f} m/s")

    # draw
    fig.canvas.draw_idle()
    plt.pause(0.001)

    # Save prevs
    _prev_posA[:] = posA
    _prev_posB[:] = posB
    _prev_A_cmd[:] = A_cmd
    _prev_B_cmd[:] = B_cmd
    _prev_sample_time = sim_time





# Warn if contact buffer saturates
def warn_if_contact_saturated():
    if data.ncon >= model.nconmax:
        print("[warn] data.ncon reached model.nconmax; increase <size nconmax=...> in XML", flush=True)

# ------------------- vectorized min-distance -------------------
# Capsules are aligned with local X → first column of geom_xmat: [0,3,6]
def min_surface_gap_and_pair():
    # Endpoints for all RA segments
    pos_RA    = data.geom_xpos[RA_idx]                       # (nRA, 3)
    xaxis_RA  = data.geom_xmat[RA_idx][:, [0, 3, 6]]         # (nRA, 3)
    hl_RA     = model.geom_size[RA_idx, 1]                   # (nRA,)
    a1        = pos_RA - hl_RA[:, None] * xaxis_RA           # (nRA, 3)
    a2        = pos_RA + hl_RA[:, None] * xaxis_RA           # (nRA, 3)

    # Endpoints for all RB segments
    pos_RB    = data.geom_xpos[RB_idx]                       # (nRB, 3)
    xaxis_RB  = data.geom_xmat[RB_idx][:, [0, 3, 6]]         # (nRB, 3)
    hl_RB     = model.geom_size[RB_idx, 1]                   # (nRB,)
    b1        = pos_RB - hl_RB[:, None] * xaxis_RB           # (nRB, 3)
    b2        = pos_RB + hl_RB[:, None] * xaxis_RB           # (nRB, 3)

    # Vectors along segments
    u = a2 - a1                                              # (nRA, 3)
    v = b2 - b1                                              # (nRB, 3)

    # Dot products (broadcasted)
    A = (u * u).sum(axis=1) + 1e-12                          # (nRA,)
    C = (v * v).sum(axis=1) + 1e-12                          # (nRB,)
    B = u @ v.T                                              # (nRA, nRB)

    u_dot_a1 = (u * a1).sum(axis=1)                          # (nRA,)
    u_dot_b1 = u @ b1.T                                      # (nRA, nRB)
    D = u_dot_a1[:, None] - u_dot_b1                         # (nRA, nRB)

    v_dot_b1 = (v * b1).sum(axis=1)                          # (nRB,)
    a1_dot_v = a1 @ v.T                                      # (nRA, nRB)
    E = a1_dot_v - v_dot_b1[None, :]                         # (nRA, nRB)

    denom = A[:, None] * C[None, :] - B**2 + 1e-12
    t = np.clip((B * E - C[None, :] * D) / denom, 0.0, 1.0)  # (nRA, nRB)
    s = np.clip((A[:, None] * E - B * D) / denom, 0.0, 1.0)  # (nRA, nRB)

    pa = a1[:, None, :] + t[..., None] * u[:, None, :]       # (nRA, nRB, 3)
    pb = b1[None, :, :] + s[..., None] * v[None, :, :]       # (nRA, nRB, 3)
    diff = pa - pb
    dist = np.linalg.norm(diff, axis=2)                      # (nRA, nRB)

    # Surface gap = center distance - (rA + rB)
    gap_matrix = dist - (RA_r0[:, None] + RB_r0[None, :])    # (nRA, nRB)
    k = int(np.argmin(gap_matrix))
    i, j = np.unravel_index(k, gap_matrix.shape)
    return float(gap_matrix[i, j]), (int(RA_idx[i]), int(RB_idx[j]))

# ------------------- knobs -------------------
TOL         = 0.0001   # m; treat gap<=TOL as "touch/near-touch" for highlight
THICK_SCALE = 1.1     # visual thickness multiplier
PRINT_DT    = 0.1    # s; how often to print to console

# ------------------- endpoint movers (manual targets with smooth ramp) -------------------
def _act_id(name):
    try:
        return model.actuator(name=name).id
    except Exception:
        return None

# Resolve actuator ids once
_RA_x = _act_id("RA_x"); _RA_y = _act_id("RA_y"); _RA_z = _act_id("RA_z")
_RB_x = _act_id("RB_x"); _RB_y = _act_id("RB_y"); _RB_z = _act_id("RB_z")

# Helper to read current joint qpos (useful as starting command)
def _joint_qpos_vec(prefix):
    def _q(jn):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_point_{jn}")
        if jid < 0: return 0.0
        return float(data.qpos[model.jnt_qposadr[jid]])
    return np.array([_q('x'), _q('y'), _q('z')], dtype=float)

# State for the ramp
XYZ_CLAMP = np.array(((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)), dtype=float)  # workspace bounds
A_cmd    = _joint_qpos_vec('A')   # last commanded A position
B_cmd    = _joint_qpos_vec('B')   # last commanded B position
A_target = A_cmd.copy()
B_target = B_cmd.copy()
A_speed  = 10.0    # m/s per-axis cap (default)
B_speed  = 10.0

def move_A_to(x=None, y=None, z=None, speed_mps=None):
    """Set A's target (absolute XYZ in meters) and optional ramp speed (m/s)."""
    global A_target, A_speed
    if x is not None: A_target[0] = float(x)
    if y is not None: A_target[1] = float(y)
    if z is not None: A_target[2] = float(z)
    if speed_mps is not None: A_speed = float(speed_mps)

def move_B_to(x=None, y=None, z=None, speed_mps=None):
    """Set B's target (absolute XYZ in meters) and optional ramp speed (m/s)."""
    global B_target, B_speed
    if x is not None: B_target[0] = float(x)
    if y is not None: B_target[1] = float(y)
    if z is not None: B_target[2] = float(z)
    if speed_mps is not None: B_speed = float(speed_mps)

def step_movers(dt):
    """Advance A/B commands toward their targets by at most (speed*dt) per axis and write to ctrl."""
    global A_cmd, B_cmd
    # clamp targets to workspace
    At = np.minimum(np.maximum(A_target, XYZ_CLAMP[:,0]), XYZ_CLAMP[:,1])
    Bt = np.minimum(np.maximum(B_target, XYZ_CLAMP[:,0]), XYZ_CLAMP[:,1])

    # per-axis ramp (independent caps)
    a_max = A_speed * dt
    b_max = B_speed * dt
    A_cmd = A_cmd + np.clip(At - A_cmd, -a_max, a_max)
    B_cmd = B_cmd + np.clip(Bt - B_cmd, -b_max, b_max)

    # write to actuators (skip if any actuator is missing)
    if _RA_x is not None: data.ctrl[_RA_x] = A_cmd[0]
    if _RA_y is not None: data.ctrl[_RA_y] = A_cmd[1]
    if _RA_z is not None: data.ctrl[_RA_z] = A_cmd[2]
    if _RB_x is not None: data.ctrl[_RB_x] = B_cmd[0]
    if _RB_y is not None: data.ctrl[_RB_y] = B_cmd[1]
    if _RB_z is not None: data.ctrl[_RB_z] = B_cmd[2]


# Slower, speed-limited circles that start at t >= start_t.
# First point matches A/B positions at start_t.
def circle_from_35(t, *, start_t=35.0, R=0.6, period_s=16.0,
                   cwA=True, cwB=False, speed_mps=1.0,
                   v_ref_max=None, ramp_s=3.0):
    """
    period_s : circle period (bigger = slower). omega = 2π/period_s
    v_ref_max: optional cap on reference tangential speed (m/s). If set,
               the reference will never move faster than v_ref_max along the circle.
    ramp_s   : seconds to ramp omega from 0 -> nominal after start_t (smooth start).
    """
    if t < start_t:
        # (optional) reset if you re-run before start
        if hasattr(circle_from_35, "_init"):
            delattr(circle_from_35, "_init")
        return

    # one-time init at first call after start_t
    if not hasattr(circle_from_35, "_init"):
        A0 = A_cmd.copy(); B0 = B_cmd.copy()
        mid = 0.1 * (A0 + B0)

        def center_and_phase(P0):
            v = mid - P0
            u = v / (np.linalg.norm(v) + 1e-12)   # unit ray toward midpoint
            c = P0 + R * u                        # center so |P0 - c| = R
            phi = np.arctan2(P0[1]-c[1], P0[0]-c[0])  # phase so τ=0 hits P0
            return c, phi, P0[2]

        circle_from_35.cA, circle_from_35.phiA, circle_from_35.zA = center_and_phase(A0)
        circle_from_35.cB, circle_from_35.phiB, circle_from_35.zB = center_and_phase(B0)
        circle_from_35.phase = 0.0
        circle_from_35.last_t = t
        circle_from_35._init = True

    # time step and ramped angular speed
    dt = max(1e-6, t - circle_from_35.last_t); circle_from_35.last_t = t
    omega_nom = 2*np.pi / float(period_s)
    ramp = np.clip((t - start_t) / max(1e-6, ramp_s), 0.0, 1.0)
    omega_eff = omega_nom * ramp

    # cap reference tangential speed if requested
    if v_ref_max is not None:
        omega_eff = min(omega_eff, float(v_ref_max) / max(1e-6, R))

    # advance phase with the effective omega
    circle_from_35.phase += omega_eff * dt

    sA = -1.0 if cwA else +1.0   # clockwise = negative sign
    sB = -1.0 if cwB else +1.0

    aθ = sA*circle_from_35.phase + circle_from_35.phiA
    bθ = sB*circle_from_35.phase + circle_from_35.phiB

    ax = circle_from_35.cA[0] + R*np.cos(aθ)
    ay = circle_from_35.cA[1] + R*np.sin(aθ)
    bx = circle_from_35.cB[0] + R*np.cos(bθ)
    by = circle_from_35.cB[1] + R*np.sin(bθ)

    move_A_to(x=ax, y=ay, z=circle_from_35.zA, speed_mps=speed_mps)
    move_B_to(x=bx, y=by, z=circle_from_35.zB, speed_mps=speed_mps)

# --- robot pose helpers -------------------------------------------------------
def get_robot_positions():
    """
    Returns a dict with world positions of the robot bodies:
      {'A_point': np.array([x, y, z]),
       'B_point': np.array([x, y, z])}
    """
    ida = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A_point")
    idb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_point")
    if ida < 0 or idb < 0:
        raise RuntimeError("Body 'A_point' or 'B_point' not found in the model.")
    # data.xpos is (nbody, 3): world position of each body's frame origin
    return {
        "A_point": data.xpos[ida].copy(),
        "B_point": data.xpos[idb].copy(),
    }

def print_robot_positions(prefix=""):
    p = get_robot_positions()
    Ax, Ay, Az = p["A_point"]
    Bx, By, Bz = p["B_point"]
    print(f"{prefix}A_point=({Ax:.4f}, {Ay:.4f}, {Az:.4f}) | "
          f"B_point=({Bx:.4f}, {By:.4f}, {Bz:.4f})", flush=True)



# ---------- in-viewer velocity arrows (actual=green, commanded=blue) ----------
_vel_prev_time = None
_vel_prev_posA = None
_vel_prev_posB = None
_vel_prev_Acmd = None
_vel_prev_Bcmd = None

def _add_arrow_to_scene(scn, p0, p1, rgba=(0,1,0,1), width=0.01):
    """Append one arrow geom from p0 -> p1 into user scene."""

    # pick the right function name for your MuJoCo build
    _connector = getattr(mujoco, "mjv_connector", None) or getattr(mujoco, "mjv_makeConnector", None)
    if _connector is None:
        raise RuntimeError("Could not find mjv_connector / mjv_makeConnector in mujoco module.")

    if scn.ngeom >= scn.maxgeom:
        return  # out of user-geom slots

    # Grab a free slot and initialize the geom. size/pos/mat are placeholders;
    # the connector call below will set pose and length.
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=float),           # size (unused here)
        np.zeros(3, dtype=float),           # pos  (unused here)
        np.eye(3, dtype=float).ravel(),     # mat  (unused here)
        np.asarray(rgba, dtype=float),
    )

    # Set the arrow to connect p0 -> p1 with given shaft width
    _connector(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    )

    scn.ngeom += 1


def draw_velocity_arrows(viewer, model, data, A_cmd, B_cmd):
    """Compute actual + commanded velocities and draw arrows at A/B body origins."""
    global _vel_prev_time, _vel_prev_posA, _vel_prev_posB, _vel_prev_Acmd, _vel_prev_Bcmd

    ida = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A_point")
    idb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_point")
    if ida < 0 or idb < 0:
        return

    pA = data.xpos[ida].copy()
    pB = data.xpos[idb].copy()
    t  = float(data.time)

    # first call: initialize history and skip drawing (no dt yet)
    if _vel_prev_time is None:
        _vel_prev_time = t
        _vel_prev_posA = pA.copy()
        _vel_prev_posB = pB.copy()
        _vel_prev_Acmd = A_cmd.copy()
        _vel_prev_Bcmd = B_cmd.copy()
        return

    dt = max(1e-9, t - _vel_prev_time)

    # Actual world linear velocities
    vA = (pA - _vel_prev_posA) / dt
    vB = (pB - _vel_prev_posB) / dt

    # Commanded velocities from your ramped commands
    vA_cmd = (A_cmd - _vel_prev_Acmd) / dt
    vB_cmd = (B_cmd - _vel_prev_Bcmd) / dt

    # Reasonable on-screen length (m per m/s) – adjust to taste
    scale_actual  = 0.4
    scale_command = 0.4

    # wipe user geoms and draw fresh arrows
    scn = viewer.user_scn
    scn.ngeom = 0
    _add_arrow_to_scene(scn, pA, pA + scale_actual  * vA,     rgba=(0.0, 0.9, 0.0, 1.0), width=0.006)
    _add_arrow_to_scene(scn, pA, pA + scale_command * vA_cmd, rgba=(0.2, 0.45, 1.0, 1.0), width=0.004)
    _add_arrow_to_scene(scn, pB, pB + scale_actual  * vB,     rgba=(0.0, 0.9, 0.0, 1.0), width=0.006)
    _add_arrow_to_scene(scn, pB, pB + scale_command * vB_cmd, rgba=(0.2, 0.45, 1.0, 1.0), width=0.004)

    # save prevs
    _vel_prev_time = t
    _vel_prev_posA[:] = pA
    _vel_prev_posB[:] = pB
    _vel_prev_Acmd[:] = A_cmd
    _vel_prev_Bcmd[:] = B_cmd


# ------------------- run with viewer -------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]  = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = True

    # keyboard handler
    #viewer.user_key_callback = viewer_key_callback

    print("CWD:", os.getcwd())
    print("Snapshot path:", os.path.abspath(SNAP_PATH))


    t0 = time.time()
    last_print = 0.0
    last_pos_print = 0.0

    substeps = 2  # try 2–4 if you go back to 0.001 step

    while viewer.is_running():
        
        #poll_snapshot_flags()
        
        t = time.time() - t0

        # --- YOUR MANUAL COMMANDS HERE (call whenever you want to change targets) ---
        # Example: after 3s, send A somewhere else and speed it up
        # if t > 5.0:
        #     move_A_to(x=-0.95, y= 0.90, z=0.0, speed_mps=0.8)
        #     move_B_to(x=-0.90, y= -0.90, z=0.0, speed_mps=0.7)
            
        # #if t > 35.0:
        # if t > 30.0:
        #     circle_from_35(
        #         t,
        #         R=0.5,
        #         period_s=20.0,     # <-- slower (e.g., 18 s per lap)
        #         cwA=True,
        #         cwB=True,
        #         speed_mps=0.8,     # <-- slower ramp to match your actuators
        #         v_ref_max=0.25,    # <-- (optional) limit reference speed to 0.25 m/s
        #         ramp_s=0.5         # <-- ease in over 4 s
        #     )
            
            
        # #if t > 50.0:
        # if t > 40.0:
        #     move_A_to(x=-0.5, y= 0.3, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.5, y= -0.3, z=0.0, speed_mps=0.5)
            
        # if t > 70.0:
        #     move_A_to(x=-0.2, y= 0.4, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.2, y= -0.4, z=0.0, speed_mps=0.5)
            
        # if t > 90.0:
        #     move_A_to(x=-0.5, y= 0.2, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.5, y= -0.2, z=0.0, speed_mps=0.5)
            
        # if t > 120.0:
        #     move_A_to(x=-0.2, y= 0.45, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.2, y= -0.45, z=0.0, speed_mps=0.5)

        # if t > 150.0:
        #     move_A_to(x=-0.5, y= 0.2, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.5, y= -0.2, z=0.0, speed_mps=0.5)

        # if t > 180.0:
        #     move_A_to(x=-0.2, y= 0.45, z=0.0, speed_mps=0.5)
        #     move_B_to(x=-0.2, y= -0.45, z=0.0, speed_mps=0.5)

        # if t > 210.0:
        #     move_A_to(x=-0.5, y= 0.2, z=0.0, speed_mps=0.6)
        #     move_B_to(x=-0.5, y= -0.2, z=0.0, speed_mps=0.6)


        # Smoothly move toward the current targets
        step_movers(model.opt.timestep)

        # step sim
        mujoco.mj_step(model, data)
        
        # # --- update live plots INPUTS every PLOT_DT_SIM seconds of sim time
        # sim_time = float(data.time)
        # if sim_time - _last_plot_sim_time >= PLOT_DT_SIM:
        #     _update_live_plots(sim_time, A_cmd, B_cmd)
        #     _last_plot_sim_time = sim_time


        # compute min surface gap and closest pair (vectorized & fast)
        gap, pair = min_surface_gap_and_pair()

        # ---------- console print ----------
        if t - last_print >= PRINT_DT:
            status = "TOUCH" if gap <= 0.0 else ("NEAR" if gap <= TOL else "CLEAR")
            a, b = (gname(pair[0]), gname(pair[1])) if pair else ("-", "-")
            # print(f"[t={t:7.3f}s] gap={gap: .5f} m  status={status}  pair={a} <-> {b}", flush=True)
            last_print = t
        # -----------------------------------


        # PRINT ROBOT POSITIONS
        if t - last_pos_print >= 0.1:
            print_robot_positions(prefix=f"[t={t:6.2f}s] ")
            last_pos_print = t

        # visual highlight: restore, then thicken (and color if allowed)
        model.geom_size[:, 0] = BASE_RADIUS
        model.geom_rgba[:]    = BASE_RGBA
        if pair and gap <= TOL:
            g1, g2 = pair
            model.geom_size[g1, 0] = BASE_RADIUS[g1] * THICK_SCALE
            model.geom_size[g2, 0] = BASE_RADIUS[g2] * THICK_SCALE
            model.geom_rgba[g1] = np.array([0.0, 1.0, 0.0, 1.0])
            model.geom_rgba[g2] = np.array([0.0, 1.0, 0.0, 1.0])

        # draw in-viewer arrows every frame
        draw_velocity_arrows(viewer, model, data, A_cmd, B_cmd)
        
        # if 150.0 < t < 150.1:
        #     save_snapshot(model, data)

        viewer.sync()
        time.sleep(0.001)
