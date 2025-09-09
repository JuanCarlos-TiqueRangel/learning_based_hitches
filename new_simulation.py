import os, time, numpy as np
os.environ.setdefault("MUJOCO_GL", "glfw")  # Windows/Conda: ensure a working GL backend

import mujoco
import mujoco.viewer

MODEL = "./model/plugin/elasticity/RLhitches.xml"
SNAP_PATH = "hitch_snapshot.npz"   # existing snapshot

# ---------------- model/data ----------------
model = mujoco.MjModel.from_xml_path(MODEL)
data  = mujoco.MjData(model)

def _copy(dst, arr, name):
    if arr is None:
        print(f"  - {name}: missing")
        return
    n = min(dst.size, arr.size)
    if dst.size != arr.size:
        print(f"  - {name}: size mismatch (dst={dst.size}, file={arr.size}); copying first {n}")
    dst.flat[:n] = arr.flat[:n]

def print_model_sizes(m):
    print("[model] sizes: nq=%d nv=%d nu=%d na=%d ngeom=%d" %
          (m.nq, m.nv, m.nu, m.na, m.ngeom))

def restore_snapshot_keep_ctrl(model, data, path):
    if not os.path.exists(path):
        raise FileNotFoundError(os.path.abspath(path))
    Z = np.load(path, allow_pickle=False)

    # Show what is inside the file
    print("[snapshot] file contents:")
    for k in ["qpos","qvel","ctrl","act","plugin_state","qacc_warmstart"]:
        if k in Z:
            print(f"  - {k}: {Z[k].shape}")
        else:
            print(f"  - {k}: <absent>")

    # Check sizes match the *current* model
    print_model_sizes(model)
    assert model.nq == Z["qpos"].size, f"qpos size mismatch: model.nq={model.nq} vs file={Z['qpos'].size}"
    assert model.nv == Z["qvel"].size, f"qvel size mismatch: model.nv={model.nv} vs file={Z['qvel'].size}"
    if "ctrl" in Z:
        assert model.nu == Z["ctrl"].size, f"ctrl size mismatch: model.nu={model.nu} vs file={Z['ctrl'].size}"

    # Reset to a clean state first
    mujoco.mj_resetData(model, data)

    # Restore pose first
    _copy(data.qpos, Z.get("qpos"), "qpos")
    mujoco.mj_forward(model, data)

    # Restore dynamics
    _copy(data.qvel, Z.get("qvel"), "qvel")

    # Keep the exact controls from the snapshot (do NOT zero)
    _copy(data.ctrl, Z.get("ctrl"), "ctrl")

    # Some plugins don’t expose state; if yours does, it will be non-zero length
    if hasattr(data, "plugin_state"):
        _copy(data.plugin_state, Z.get("plugin_state"), "plugin_state")

    # Clear cached impulses
    if hasattr(data, "qacc"):            data.qacc[:] = 0.0
    if hasattr(data, "qacc_warmstart"):  data.qacc_warmstart[:] = 0.0
    if hasattr(data, "efc_force"):       data.efc_force[:] = 0.0
    if hasattr(data, "efc_state"):       data.efc_state[:] = 0
    if hasattr(data, "qfrc_constraint"): data.qfrc_constraint[:] = 0.0
    if hasattr(data, "qfrc_applied"):    data.qfrc_applied[:] = 0.0

    mujoco.mj_forward(model, data)
    data.time = 0.0

def soft_settle_contacts(model, data,
                         settle_dt=1e-4,
                         settle_time=1.0,
                         damping_mult=12.0,
                         margin_base=0.006,
                         fric_scale=0.15):
    """Keep contacts ON but make them gentle for a short settle window."""
    opt = model.opt

    # Save originals
    orig = dict(
        timestep=opt.timestep,
        solimp=opt.o_solimp.copy(),
        solref=opt.o_solref.copy(),
        iterations=int(opt.iterations),
        tolerance=float(opt.tolerance),
        noslip_iterations=int(opt.noslip_iterations),
        disableflags=int(opt.disableflags),
        dof_damping=model.dof_damping.copy(),
        geom_margin=model.geom_margin.copy(),
        geom_friction=model.geom_friction.copy(),
    )

    # Gentle settings (contacts ON)
    opt.timestep = float(settle_dt)
    opt.o_solimp[:] = np.array([0.99, 0.9999, 0.0002, 0.5, 2.0])
    opt.o_solref[:] = np.array([0.01, 1.0])
    opt.iterations  = max(200, orig["iterations"])
    opt.tolerance   = min(1e-6, orig["tolerance"])
    opt.noslip_iterations = max(40, orig["noslip_iterations"])
    try:
        opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    except Exception:
        opt.disableflags |= (1 << 4)

    # Extra damping & larger margins reduce first impulses
    model.dof_damping[:] = np.maximum(orig["dof_damping"] * damping_mult, 1e-6)
    model.geom_margin[:] = np.maximum(orig["geom_margin"], margin_base)
    model.geom_friction[:] = orig["geom_friction"]
    model.geom_friction[:, 0] *= fric_scale
    model.geom_friction[:, 1] *= fric_scale
    model.geom_friction[:, 2] *= fric_scale

    # Settle
    steps = int(max(1, settle_time / opt.timestep))
    for i in range(steps):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("Non-finite state encountered during settle")

    # Restore originals
    opt.o_solimp[:] = orig["solimp"]
    opt.o_solref[:] = orig["solref"]
    opt.timestep    = orig["timestep"]
    opt.iterations  = orig["iterations"]
    opt.tolerance   = orig["tolerance"]
    opt.noslip_iterations = orig["noslip_iterations"]
    opt.disableflags = orig["disableflags"]
    model.dof_damping[:]  = orig["dof_damping"]
    model.geom_margin[:]  = orig["geom_margin"]
    model.geom_friction[:] = orig["geom_friction"]

    mujoco.mj_forward(model, data)

# ---------- load & (try to) settle ----------
try:
    restore_snapshot_keep_ctrl(model, data, SNAP_PATH)
    try:
        soft_settle_contacts(model, data,
                             settle_dt=1e-4,
                             settle_time=1.2,
                             damping_mult=14.0,
                             margin_base=0.006,
                             fric_scale=0.12)
        print(f"[snapshot] restored & settled from {os.path.abspath(SNAP_PATH)}")
    except Exception as e:
        print("[settle] WARNING:", e)
        print("         Proceeding without settle so you can inspect the state.")
        mujoco.mj_forward(model, data)
except AssertionError as e:
    print("[snapshot] SIZE MISMATCH →", e)
    print("Make sure the XML is *exactly* the one used when saving this .npz.")
except Exception as e:
    print("[snapshot] load failed:", e)

# ---------- run viewer regardless ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
