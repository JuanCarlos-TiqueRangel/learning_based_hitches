import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "./model/plugin/elasticity/RLhitches.xml"

# ------------------- load -------------------
model = mujoco.MjModel.from_xml_path(MODEL)
data  = mujoco.MjData(model)

def gname(g):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

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

# ---- make hitches hold: frictional contact on cable capsules ----
ids = np.array(RA + RB, dtype=int)
model.geom_matid[ids]   = -1          # ensure per-geom colors can show
model.geom_condim[ids]  = 6           # enable sliding + torsional + rolling friction
model.geom_friction[ids, 0] = 0.1     # slide
model.geom_friction[ids, 1] = 0.005   # spin
model.geom_friction[ids, 2] = 0.0001  # roll
model.geom_margin[ids]  = 0.0005      # small positive margin helps stability

# Optional global contact softening (stability)
model.opt.o_solimp[:] = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
model.opt.o_solref[:] = np.array([0.02, 1.0])
model.opt.timestep    = 0.001  # tighter time step during knotting

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
A_speed  = 0.6    # m/s per-axis cap (default)
B_speed  = 0.6

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













# ------------------- run with viewer -------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]  = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = True

    t0 = time.time()
    last_print = 0.0

    # EXAMPLE: set initial targets once (you can change these anytime)
    move_A_to(x=0.10, y= 0.50, z=0.5, speed_mps=0.8)
    move_B_to(x=0.10, y= 0.50, z=0.5, speed_mps=0.8)

    while viewer.is_running():
        t = time.time() - t0

        # --- YOUR MANUAL COMMANDS HERE (call whenever you want to change targets) ---
        # Example: after 3s, send A somewhere else and speed it up
        print("[DEBUG] TIME: ", t)
        if t > 15.0:
            move_A_to(x=-0.95, y= 0.90, z=0.0, speed_mps=3.5)
            move_B_to(x=-0.92, y= -0.90, z=0.0, speed_mps=3.5)
            
        #if t > 35.0:
        if t > 200.0:
            circle_from_35(
                t,
                R=0.5,
                period_s=22.0,     # <-- slower (e.g., 18 s per lap)
                cwA=True,
                cwB=True,
                speed_mps=0.8,     # <-- slower ramp to match your actuators
                v_ref_max=0.25,    # <-- (optional) limit reference speed to 0.25 m/s
                ramp_s=4.0         # <-- ease in over 4 s
            )
            
            
        #if t > 50.0:
        if t > 300.0:
            move_A_to(x=-0.7, y= -0.90, z=0.0, speed_mps=1.5)
            move_B_to(x=-0.7, y= 0.90, z=0.0, speed_mps=1.5)





            # move_A_to(x=0.55, y= 0.90, z=0.0, speed_mps=3.5)
            # move_B_to(x=-0.52, y= -0.90, z=0.0, speed_mps=3.5)
            
            
        # if t > 55.0:
        #     move_A_to(x= 0.55, y= 0.90, z=0.0, speed_mps=0.0)
        #     move_B_to(x=-0.92, y= 0.0, z=0.5, speed_mps=3.5)
            
        # if t > 75.0:
        #     move_A_to(x= 0.55, y= 0.90, z=0.0, speed_mps=0.0)
        #     move_B_to(x=-0.92, y= 0.0, z=-0.5, speed_mps=1.5)
            
        # if t > 95.0:
        #     move_A_to(x= 0.0, y= 0.90, z=0.0, speed_mps=0.0)
        #     move_B_to(x=-1.0, y= -0.5, z=-0.5, speed_mps=1.5)
            
        # if t > 115.0:
        #     move_A_to(x= 0.0, y= 0.90, z=0.0, speed_mps=0.0)
        #     move_B_to(x=-1.0, y= -0.7, z=0.0, speed_mps=1.5)
            
        # if t > 130.0:
        #     move_A_to(x= 1.0, y= 0.5, z=0.0, speed_mps=2.0)
        #     move_B_to(x=-1.0, y= 1.0, z=0.0, speed_mps=2.0)
            
        # # if t > 145.0:
        # #     move_A_to(x= -1.0, y= -1.0, z=0.0, speed_mps=2.0)
        # #     move_B_to(x=-1.0, y= 1.0, z=0.0, speed_mps=2.0)
            
        # # if t > 175.0:
        # #     move_A_to(x= -0.5, y= -1.0, z=0.5, speed_mps=0.5)
        # #     move_B_to(x=-1.5, y= 1.0, z=0.5, speed_mps=0.5)
            
        # #if t > 200.0:
        # if t > 145.0:
        #     move_A_to(x= 0.6, y= -0.5, z= 0.5, speed_mps=0.5)
        #     move_B_to(x= 0.6, y= 0.5, z= 0.5, speed_mps=0.5)
            
        # if t > 170.0:
        #     move_A_to(x= 0.2, y= 0.3, z= 0.5, speed_mps=0.5)
        #     move_B_to(x= 0.2, y= -0.5, z= 0.5, speed_mps=0.5)

        # Keep an eye on contact buffer
        #warn_if_contact_saturated()

        # Smoothly move toward the current targets
        step_movers(model.opt.timestep)

        # step sim
        mujoco.mj_step(model, data)

        # compute min surface gap and closest pair (vectorized & fast)
        gap, pair = min_surface_gap_and_pair()

        # ---------- console print ----------
        if t - last_print >= PRINT_DT:
            status = "TOUCH" if gap <= 0.0 else ("NEAR" if gap <= TOL else "CLEAR")
            a, b = (gname(pair[0]), gname(pair[1])) if pair else ("-", "-")
            # print(f"[t={t:7.3f}s] gap={gap: .5f} m  status={status}  pair={a} <-> {b}", flush=True)
            last_print = t
        # -----------------------------------

        # visual highlight: restore, then thicken (and color if allowed)
        model.geom_size[:, 0] = BASE_RADIUS
        model.geom_rgba[:]    = BASE_RGBA
        if pair and gap <= TOL:
            g1, g2 = pair
            model.geom_size[g1, 0] = BASE_RADIUS[g1] * THICK_SCALE
            model.geom_size[g2, 0] = BASE_RADIUS[g2] * THICK_SCALE
            model.geom_rgba[g1] = np.array([0.0, 1.0, 0.0, 1.0])
            model.geom_rgba[g2] = np.array([0.0, 1.0, 0.0, 1.0])

        viewer.sync()
        time.sleep(0.001)
