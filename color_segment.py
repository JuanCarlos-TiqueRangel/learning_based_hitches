
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "./model/plugin/elasticity/RLhitches.xml"
SEG_NAMES = ["RAG30", "RAG30", "RAG40", "RAG50"]   # put your segment names here

GREEN = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
RED   = np.array([1.0, 0.1, 0.1, 1.0], dtype=float)

model = mujoco.MjModel.from_xml_path(MODEL)
data  = mujoco.MjData(model)

# Resolve geoms and record their current material (if any)
geom_ids = []
for name in SEG_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        print(f"[warn] Geom '{name}' not found; skipping")
        continue
    geom_ids.append(gid)

if not geom_ids:
    raise RuntimeError("No valid geoms resolved from SEG_NAMES.")

# Collect materials used by these geoms (some plugins force a material each step)
mat_ids = []
for gid in geom_ids:
    matid = int(model.geom_matid[gid])
    if matid >= 0:
        mat_ids.append(matid)
    else:
        # detach any material so per-geom rgba is used (if plugin doesn't override later)
        model.geom_matid[gid] = -1
        model.geom_rgba[gid] = GREEN

# Deduplicate
mat_ids = sorted(set(mat_ids))

print(f"[info] controlling {len(geom_ids)} geoms; {len(mat_ids)} materials involved: {mat_ids}")

#with mujoco.viewer.launch(model, data) as viewer:
with mujoco.viewer.launch_passive(model, data) as viewer:
    period = 1.0
    next_change = time.monotonic() + period
    toggle = False
    t0 = time.monotonic()

    while viewer.is_running():
        # (optional) tiny control to show sim is alive
        if model.nu > 0:
            data.ctrl[0] = 0.2 * np.sin(2*np.pi*0.5*(time.monotonic() - t0))

        mujoco.mj_step(model, data)

        # ---- live color change every second ----
        now = time.monotonic()
        if now >= next_change:
            toggle = not toggle
            col = RED if toggle else GREEN

            # 1) If plugin reassigns a material, recolor the material (affects all geoms using it)
            for mid in mat_ids:
                # guard: mid might change if plugin swaps materials; re-read from geoms:
                # (rebuild mat set from current geoms to survive plugin rewrites)
                pass
            # rebuild material set from current geoms each time (robust to plugin resets)
            cur_mats = set()
            for gid in geom_ids:
                cur_mats.add(int(model.geom_matid[gid]))
            cur_mats.discard(-1)  # only real materials
            for mid in cur_mats:
                model.mat_rgba[mid] = col

            # 2) For geoms without materials (or if you forced matid = -1), recolor per-geom
            for gid in geom_ids:
                if int(model.geom_matid[gid]) == -1:
                    model.geom_rgba[gid] = col

            # (optional) print debug info so you SEE it's live + catch plugin overwrites
            sample_gid = geom_ids[0]
            print(
                f"{now - t0:6.2f}s  color -> {'RED' if toggle else 'GREEN'} | "
                f"gid {sample_gid}: matid={int(model.geom_matid[sample_gid])}, "
                f"geom_rgba={model.geom_rgba[sample_gid]}, "
                f"mat_rgba(first)={(model.mat_rgba[int(model.geom_matid[sample_gid])].tolist() if int(model.geom_matid[sample_gid])!=-1 else 'NA')}"
            )

            next_change += period

        viewer.sync()
        time.sleep(0.001)
