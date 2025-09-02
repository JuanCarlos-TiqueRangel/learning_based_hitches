# import time
# import numpy as np
# import mujoco
# import mujoco.viewer

# MODEL = "./model/plugin/elasticity/RLhitches.xml"

# # ------------------- load -------------------
# model = mujoco.MjModel.from_xml_path(MODEL)
# data  = mujoco.MjData(model)

# def gname(g):
#     return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

# # collect the capsule geoms that belong to each cable (by name)
# RA = [g for g in range(model.ngeom)
#       if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and "RAG" in gname(g)]
# RB = [g for g in range(model.ngeom)
#       if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and "RBG" in gname(g)]
# print(f"[collect] RA={len(RA)}  RB={len(RB)}", flush=True)
# if not RA or not RB:
#     raise RuntimeError("Could not find RAG*/RBG* capsule geoms. Check names / XML.")

# # allow per-geom colors if the plugin isn't overriding them (material off)
# for g in RA + RB:
#     model.geom_matid[g] = -1




# # precompute index arrays and base radii (so runtime thickness changes don't affect gap math)
# RA_idx = np.array(RA, dtype=int)
# RB_idx = np.array(RB, dtype=int)
# BASE_RADIUS = model.geom_size[:, 0].copy()       # keep originals to restore each frame (visual)
# BASE_RGBA   = model.geom_rgba.copy()
# RA_r0 = BASE_RADIUS[RA_idx].copy()               # use base radii for gap computation
# RB_r0 = BASE_RADIUS[RB_idx].copy()


# # ---- make hitches hold: frictional contact on cable capsules ----
# ids = np.array(RA + RB, dtype=int)
# model.geom_matid[ids]   = -1          # ensure per-geom colors can show
# model.geom_condim[ids]  = 6           # enable sliding + torsional + rolling friction
# model.geom_friction[ids, 0] = 1.2     # slide
# model.geom_friction[ids, 1] = 0.005   # spin
# model.geom_friction[ids, 2] = 0.0001  # roll
# model.geom_margin[ids]  = 0.0005      # small positive margin helps stability

# # Optional global contact softening (stability)
# model.opt.o_solimp[:] = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
# model.opt.o_solref[:] = np.array([0.02, 1.0])
# model.opt.timestep    = 0.001  # tighter time step during knotting

# # Warn if contact buffer saturates
# def warn_if_contact_saturated():
#     if data.ncon >= model.nconmax:
#         print("[warn] data.ncon reached model.nconmax; increase <size nconmax=...> in XML", flush=True)


# # ------------------- vectorized min-distance -------------------
# # Capsules are aligned with local X → first column of geom_xmat: [0,3,6]
# def min_surface_gap_and_pair():
#     # Endpoints for all RA segments
#     pos_RA    = data.geom_xpos[RA_idx]                       # (nRA, 3)
#     xaxis_RA  = data.geom_xmat[RA_idx][:, [0, 3, 6]]         # (nRA, 3)
#     hl_RA     = model.geom_size[RA_idx, 1]                   # (nRA,)
#     a1        = pos_RA - hl_RA[:, None] * xaxis_RA           # (nRA, 3)
#     a2        = pos_RA + hl_RA[:, None] * xaxis_RA           # (nRA, 3)

#     # Endpoints for all RB segments
#     pos_RB    = data.geom_xpos[RB_idx]                       # (nRB, 3)
#     xaxis_RB  = data.geom_xmat[RB_idx][:, [0, 3, 6]]         # (nRB, 3)
#     hl_RB     = model.geom_size[RB_idx, 1]                   # (nRB,)
#     b1        = pos_RB - hl_RB[:, None] * xaxis_RB           # (nRB, 3)
#     b2        = pos_RB + hl_RB[:, None] * xaxis_RB           # (nRB, 3)

#     # Vectors along segments
#     u = a2 - a1                                              # (nRA, 3)
#     v = b2 - b1                                              # (nRB, 3)

#     # Dot products (broadcasted)
#     A = (u * u).sum(axis=1) + 1e-12                          # (nRA,)
#     C = (v * v).sum(axis=1) + 1e-12                          # (nRB,)
#     B = u @ v.T                                              # (nRA, nRB)

#     u_dot_a1 = (u * a1).sum(axis=1)                          # (nRA,)
#     u_dot_b1 = u @ b1.T                                      # (nRA, nRB)
#     D = u_dot_a1[:, None] - u_dot_b1                         # (nRA, nRB)

#     v_dot_b1 = (v * b1).sum(axis=1)                          # (nRB,)
#     a1_dot_v = a1 @ v.T                                      # (nRA, nRB)
#     E = a1_dot_v - v_dot_b1[None, :]                         # (nRA, nRB)

#     denom = A[:, None] * C[None, :] - B**2 + 1e-12
#     t = np.clip((B * E - C[None, :] * D) / denom, 0.0, 1.0)  # (nRA, nRB)
#     s = np.clip((A[:, None] * E - B * D) / denom, 0.0, 1.0)  # (nRA, nRB)

#     pa = a1[:, None, :] + t[..., None] * u[:, None, :]       # (nRA, nRB, 3)
#     pb = b1[None, :, :] + s[..., None] * v[None, :, :]       # (nRA, nRB, 3)
#     diff = pa - pb
#     dist = np.linalg.norm(diff, axis=2)                      # (nRA, nRB)

#     # Surface gap = center distance - (rA + rB)
#     gap_matrix = dist - (RA_r0[:, None] + RB_r0[None, :])    # (nRA, nRB)
#     k = int(np.argmin(gap_matrix))
#     i, j = np.unravel_index(k, gap_matrix.shape)
#     return float(gap_matrix[i, j]), (int(RA_idx[i]), int(RB_idx[j]))

# # ------------------- knobs -------------------
# TOL         = 0.005    # m; treat gap<=TOL as "touch/near-touch" for highlight
# THICK_SCALE = 1.5     # visual thickness multiplier
# PRINT_DT    = 0.05    # s; how often to print to console

# # ------------------- optional endpoint motion -------------------
# def safe_act(n):
#     try: return model.actuator(name=n).id
#     except Exception: return None
# ax,ay,az = map(safe_act, ["RA_x","RA_y","RA_z"])
# bx,by,bz = map(safe_act, ["RB_x","RB_y","RB_z"])




# class EndpointPilot:
#     """
#     Simple, readable autopilot for a 3-DOF endpoint driven by position actuators.
#     - Moves in a circle in XZ by default (Y fixed unless you give y_amp > 0).
#     - Writes absolute joint targets into data.ctrl[...] for your position actuators.
#     - Applies per-axis clamps and a max speed cap for stability.
#     """

#     def __init__(self,
#                  model, data,
#                  joint_names,          # ('A_point_x','A_point_y','A_point_z') or ('B_point_x',...)
#                  actuator_names,       # ('RA_x','RA_y','RA_z') or ('RB_x','RB_y','RB_z')
#                  center=None,          # (cx,cy,cz). If None, read current joint qpos as center
#                  x_amp=0.10, z_amp=0.06, y_amp=0.00,    # meters
#                  x_hz=0.35, z_hz=0.22, y_hz=0.20,       # Hz
#                  x_phase=0.00, z_phase=0.00, y_phase=0.0,
#                  clamps=None,          # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
#                  max_speed=0.6         # m/s per-axis cap
#                  ):
#         self.model = model
#         self.data  = data

#         # Resolve actuator IDs robustly
#         def aid(name):
#             idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
#             return None if idx < 0 else idx

#         self.ax = aid(actuator_names[0])
#         self.ay = aid(actuator_names[1])
#         self.az = aid(actuator_names[2])

#         # Resolve joint qpos addresses to read current positions if center is None
#         def joint_qpos(jname):
#             jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
#             if jid < 0:
#                 return 0.0
#             adr = model.jnt_qposadr[jid]
#             return float(data.qpos[adr])

#         if center is None:
#             cx = joint_qpos(joint_names[0])
#             cy = joint_qpos(joint_names[1])
#             cz = joint_qpos(joint_names[2])
#             center = (cx, cy, cz)

#         self.center = np.array(center, dtype=float)

#         # Motion parameters
#         self.x_amp, self.z_amp, self.y_amp = float(x_amp), float(z_amp), float(y_amp)
#         self.x_hz,  self.z_hz,  self.y_hz  = float(x_hz),  float(z_hz),  float(y_hz)
#         self.x_ph,  self.z_ph,  self.y_ph  = float(x_phase), float(z_phase), float(y_phase)

#         # Safety
#         self.clamps    = None if clamps is None else np.array(clamps, dtype=float)  # shape (3,2)
#         self.dt        = float(model.opt.timestep)
#         self.max_delta = float(max_speed) * self.dt  # max per-step change in meters

#         # Start from center
#         self.prev = self.center.copy()

#     def _target_xyz(self, t):
#         """Circle in XZ, optional sine on Y."""
#         two_pi_t = 2.0 * np.pi * t
#         # x = self.center[0] + self.x_amp * np.cos(two_pi_t * self.x_hz + self.x_ph)
#         # z = self.center[2] + self.z_amp * np.sin(two_pi_t * self.z_hz + self.z_ph)
#         # y = self.center[1] + self.y_amp * np.sin(two_pi_t * self.y_hz + self.y_ph)
#         x = 0.1
#         y = 0.1
#         z = 0.1
#         return np.array([x, y, z], dtype=float)

#     def step(self, t):
#         """Compute smooth target at time t and write to data.ctrl (position actuators)."""
#         tgt = self._target_xyz(t)

#         print(" ")
#         print(tgt)
#         print(" ")

#         # Clamp to workspace if provided
#         if self.clamps is not None:
#             tgt = np.minimum(np.maximum(tgt, self.clamps[:, 0]), self.clamps[:, 1])

#         # Per-axis speed cap to avoid jumps if dt jitters
#         delta = np.clip(tgt - self.prev, -self.max_delta, self.max_delta)
#         cmd = self.prev + delta

#         # Write to actuators (skip any that are missing)
#         if self.ax is not None: self.data.ctrl[self.ax] = cmd[0]
#         if self.ay is not None: self.data.ctrl[self.ay] = cmd[1]
#         if self.az is not None: self.data.ctrl[self.az] = cmd[2]

#         self.prev[:] = cmd




# # Reasonable workspace bounds for your scene (tweak if needed)
# XYZ_CLAMP = ((0.3, 1.1), (-0.8, 0.8), (0.6, 1.6))

# pilot_A = EndpointPilot(
#     model, data,
#     joint_names=('A_point_x','A_point_y','A_point_z'),
#     actuator_names=('RA_x','RA_y','RA_z'),
#     x_amp=0.10, z_amp=0.06, y_amp=0.0,
#     x_hz=0.35,  z_hz=0.22,  y_hz=0.20,
#     x_phase=0.0, z_phase=0.0, y_phase=0.0,
#     clamps=XYZ_CLAMP, max_speed=0.6
# )

# pilot_B = EndpointPilot(
#     model, data,
#     joint_names=('B_point_x','B_point_y','B_point_z'),
#     actuator_names=('RB_x','RB_y','RB_z'),
#     x_amp=0.10, z_amp=0.06, y_amp=0.0,
#     x_hz=0.35,  z_hz=0.22,  y_hz=0.20,
#     #x_phase=1.57, z_phase=1.57, y_phase=0.00,  # ~90° offset
#     x_phase=0.0, z_phase=0.0, y_phase=0.0,  # ~90° offset
#     clamps=XYZ_CLAMP, max_speed=0.6
# )






# # ------------------- run with viewer -------------------
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # Turn OFF heavy overlays for speed (enable if you need them)
#     # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]  = True
#     # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = True

#     t0 = time.time()
#     last_print = 0.0
#     while viewer.is_running():
#         t = time.time() - t0

#         pilot_A.step(t)
#         pilot_B.step(t)
#         #warn_if_contact_saturated()  # keep this check from your code


#         # step sim (manual manipulation works in passive viewer)
#         mujoco.mj_step(model, data)

#         # compute min surface gap and closest pair (vectorized & fast)
#         gap, pair = min_surface_gap_and_pair()

#         # ---------- console print ----------
#         if t - last_print >= PRINT_DT:
#             status = "TOUCH" if gap <= 0.0 else ("NEAR" if gap <= TOL else "CLEAR")
#             a, b = (gname(pair[0]), gname(pair[1])) if pair else ("-", "-")
#             #print(f"[t={t:7.3f}s] gap={gap: .5f} m  status={status}  pair={a} <-> {b}", flush=True)
#             last_print = t
#         # -----------------------------------

#         # visual highlight: restore, then thicken (and color if allowed)
#         model.geom_size[:, 0] = BASE_RADIUS
#         model.geom_rgba[:]    = BASE_RGBA
#         if pair and gap <= TOL:
#             g1, g2 = pair
#             model.geom_size[g1, 0] = BASE_RADIUS[g1] * THICK_SCALE
#             model.geom_size[g2, 0] = BASE_RADIUS[g2] * THICK_SCALE
#             # Color shows only if plugin isn't overriding colors (ensure vmax=0 in XML)
#             model.geom_rgba[g1] = np.array([0.0, 1.0, 0.0, 1.0])
#             model.geom_rgba[g2] = np.array([0.0, 1.0, 0.0, 1.0])

#         viewer.sync()
#         # keep a tiny sleep for UI responsiveness; increase for even lower CPU
#         time.sleep(0.001)













































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
model.geom_friction[ids, 0] = 1.2     # slide
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
TOL         = 0.005    # m; treat gap<=TOL as "touch/near-touch" for highlight
THICK_SCALE = 1.5     # visual thickness multiplier
PRINT_DT    = 0.05    # s; how often to print to console

# ------------------- optional endpoint motion -------------------
def safe_act(n):
    try: return model.actuator(name=n).id
    except Exception: return None
ax,ay,az = map(safe_act, ["RA_x","RA_y","RA_z"])
bx,by,bz = map(safe_act, ["RB_x","RB_y","RB_z"])




class EndpointPilot:
    """
    Simple, readable autopilot for a 3-DOF endpoint driven by position actuators.
    - Moves in a circle in XZ by default (Y fixed unless you give y_amp > 0).
    - Writes absolute joint targets into data.ctrl[...] for your position actuators.
    - Applies per-axis clamps and a max speed cap for stability.
    """

    def __init__(self,
                 model, data,
                 joint_names,          # ('A_point_x','A_point_y','A_point_z') or ('B_point_x',...)
                 actuator_names,       # ('RA_x','RA_y','RA_z') or ('RB_x','RB_y','RB_z')
                 center=None,          # (cx,cy,cz). If None, read current joint qpos as center
                 x_amp=0.10, z_amp=0.06, y_amp=0.0,    # meters
                 x_hz=0.35, z_hz=0.22, y_hz=0.20,       # Hz
                 x_phase=0.0, z_phase=0.0, y_phase=0.0,
                 clamps=None,          # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
                 max_speed=0.6         # m/s per-axis cap
                 ):
        self.model = model
        self.data  = data

        # Resolve actuator IDs robustly
        def aid(name):
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return None if idx < 0 else idx

        self.ax = aid(actuator_names[0])
        self.ay = aid(actuator_names[1])
        self.az = aid(actuator_names[2])

        # Resolve joint qpos addresses to read current positions if center is None
        def joint_qpos(jname):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                return 0.0
            adr = model.jnt_qposadr[jid]
            return float(data.qpos[adr])

        if center is None:
            cx = joint_qpos(joint_names[0])
            cy = joint_qpos(joint_names[1])
            cz = joint_qpos(joint_names[2])
            center = (cx, cy, cz)

        self.center = np.array(center, dtype=float)

        # Motion parameters
        self.x_amp, self.z_amp, self.y_amp = float(x_amp), float(z_amp), float(y_amp)
        self.x_hz,  self.z_hz,  self.y_hz  = float(x_hz),  float(z_hz),  float(y_hz)
        self.x_ph,  self.z_ph,  self.y_ph  = float(x_phase), float(z_phase), float(y_phase)

        # Safety
        self.clamps    = None if clamps is None else np.array(clamps, dtype=float)  # shape (3,2)
        self.dt        = float(model.opt.timestep)
        self.max_delta = float(max_speed) * self.dt  # max per-step change in meters

        # Start from center
        self.prev = self.center.copy()

    def _target_xyz(self, t):
        """Circle in XZ, optional sine on Y."""
        two_pi_t = 2.0 * np.pi * t
        x = self.center[0] + self.x_amp * np.cos(two_pi_t * self.x_hz + self.x_ph)
        z = self.center[2] + self.z_amp * np.sin(two_pi_t * self.z_hz + self.z_ph)
        y = self.center[1] + self.y_amp * np.sin(two_pi_t * self.y_hz + self.y_ph)
        return np.array([x, y, z], dtype=float)

    def step(self, t):
        """Compute smooth target at time t and write to data.ctrl (position actuators)."""
        tgt = self._target_xyz(t)

        # Clamp to workspace if provided
        if self.clamps is not None:
            tgt = np.minimum(np.maximum(tgt, self.clamps[:, 0]), self.clamps[:, 1])

        # Per-axis speed cap to avoid jumps if dt jitters
        delta = np.clip(tgt - self.prev, -self.max_delta, self.max_delta)
        cmd = self.prev + delta

        # Write to actuators (skip any that are missing)
        if self.ax is not None: self.data.ctrl[self.ax] = cmd[0]
        if self.ay is not None: self.data.ctrl[self.ay] = cmd[1]
        if self.az is not None: self.data.ctrl[self.az] = cmd[2]

        self.prev[:] = cmd




# Reasonable workspace bounds for your scene (tweak if needed)
XYZ_CLAMP = ((0.3, 1.1), (-0.8, 0.8), (0.6, 1.6))

pilot_A = EndpointPilot(
    model, data,
    joint_names=('A_point_x','A_point_y','A_point_z'),
    actuator_names=('RA_x','RA_y','RA_z'),
    x_amp=0.10, z_amp=0.06, y_amp=0.0,
    x_hz=0.35,  z_hz=0.22,  y_hz=0.20,
    x_phase=0.0, z_phase=0.0, y_phase=0.0,
    clamps=XYZ_CLAMP, max_speed=0.6
)

pilot_B = EndpointPilot(
    model, data,
    joint_names=('B_point_x','B_point_y','B_point_z'),
    actuator_names=('RB_x','RB_y','RB_z'),
    x_amp=0.10, z_amp=0.06, y_amp=0.0,
    x_hz=0.35,  z_hz=0.22,  y_hz=0.20,
    #x_phase=1.57, z_phase=1.57, y_phase=0.00,  # ~90° offset
    x_phase=0.0, z_phase=0.0, y_phase=0.0,  # ~90° offset
    clamps=XYZ_CLAMP, max_speed=0.6
)







# --- resolve actuator ids once ---
def _act_id(name):
    try:
        return model.actuator(name=name).id
    except Exception:
        return None

_RA_x = _act_id("RA_x"); _RA_y = _act_id("RA_y"); _RA_z = _act_id("RA_z")
_RB_x = _act_id("RB_x"); _RB_y = _act_id("RB_y"); _RB_z = _act_id("RB_z")

def set_A_xyz(x=None, y=None, z=None):
    """Set A endpoint absolute targets (meters) for the 3 slide joints."""
    if _RA_x is not None and x is not None: data.ctrl[_RA_x] = float(x)
    if _RA_y is not None and y is not None: data.ctrl[_RA_y] = float(y)
    if _RA_z is not None and z is not None: data.ctrl[_RA_z] = float(z)

def set_B_xyz(x=None, y=None, z=None):
    """Set B endpoint absolute targets (meters) for the 3 slide joints."""
    if _RB_x is not None and x is not None: data.ctrl[_RB_x] = float(x)
    if _RB_y is not None and y is not None: data.ctrl[_RB_y] = float(y)
    if _RB_z is not None and z is not None: data.ctrl[_RB_z] = float(z)








# ------------------- run with viewer -------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Turn OFF heavy overlays for speed (enable if you need them)
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]  = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = True

    t0 = time.time()
    last_print = 0.0
    while viewer.is_running():
        t = time.time() - t0

        # pilot_A.step(t)
        # pilot_B.step(t)
        # #warn_if_contact_saturated()  # keep this check from your code

        # --- YOUR MANUAL COORDS HERE ---
        # Example: hold A at (0.80, -0.50, 1.05) and B at (0.62, 0.50, 0.95)
        # set_A_xyz(x=0.80, y=-0.50, z=1.05)
        # set_B_xyz(x=0.62, y= 0.50, z=0.15)

        # step sim (manual manipulation works in passive viewer)
        mujoco.mj_step(model, data)

        # compute min surface gap and closest pair (vectorized & fast)
        gap, pair = min_surface_gap_and_pair()

        # ---------- console print ----------
        if t - last_print >= PRINT_DT:
            status = "TOUCH" if gap <= 0.0 else ("NEAR" if gap <= TOL else "CLEAR")
            a, b = (gname(pair[0]), gname(pair[1])) if pair else ("-", "-")
            #print(f"[t={t:7.3f}s] gap={gap: .5f} m  status={status}  pair={a} <-> {b}", flush=True)
            last_print = t
        # -----------------------------------

        # visual highlight: restore, then thicken (and color if allowed)
        model.geom_size[:, 0] = BASE_RADIUS
        model.geom_rgba[:]    = BASE_RGBA
        if pair and gap <= TOL:
            g1, g2 = pair
            model.geom_size[g1, 0] = BASE_RADIUS[g1] * THICK_SCALE
            model.geom_size[g2, 0] = BASE_RADIUS[g2] * THICK_SCALE
            # Color shows only if plugin isn't overriding colors (ensure vmax=0 in XML)
            model.geom_rgba[g1] = np.array([0.0, 1.0, 0.0, 1.0])
            model.geom_rgba[g2] = np.array([0.0, 1.0, 0.0, 1.0])

        viewer.sync()
        # keep a tiny sleep for UI responsiveness; increase for even lower CPU
        time.sleep(0.001)
