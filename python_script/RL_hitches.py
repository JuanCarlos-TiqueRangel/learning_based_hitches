import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model = mujoco.MjModel.from_xml_path("../model/plugin/elasticity/RLhitches.xml")
data = mujoco.MjData(model)

# Get actuator IDs for the red point position controllers
act_ids = {
    "RA_x": model.actuator(name="RA_x").id,
    "RA_y": model.actuator(name="RA_y").id,
    "RA_z": model.actuator(name="RA_z").id,
    "RB_x": model.actuator(name="RB_x").id,
    "RB_y": model.actuator(name="RB_y").id,
    "RB_z": model.actuator(name="RB_z").id,
}


# Define a function to set target positions for both red points
def set_target_positions(ra_xyz, rb_xyz):
    data.ctrl[act_ids["RA_x"]] = ra_xyz[0]
    data.ctrl[act_ids["RA_y"]] = ra_xyz[1]
    data.ctrl[act_ids["RA_z"]] = ra_xyz[2]

    data.ctrl[act_ids["RB_x"]] = rb_xyz[0]
    data.ctrl[act_ids["RB_y"]] = rb_xyz[1]
    data.ctrl[act_ids["RB_z"]] = rb_xyz[2]


# Open viewer and simulate
with mujoco.viewer.launch(model, data) as viewer:
    print("Simulation started. Press ESC to exit.")

    t_start = time.time()
    while viewer.is_running():
        # Elapsed time
        t = time.time() - t_start

        # Move A and B points in sine patterns
        ra_target = np.array([0.7 + 0.1 * np.sin(t), -0.5, 1.0 + 0.05 * np.sin(0.5 * t)])
        rb_target = np.array([0.7 + 0.1 * np.cos(t), 0.5, 1.0 + 0.05 * np.cos(0.5 * t)])

        # Set target positions
        set_target_positions(ra_target, rb_target)

        # Step simulation
        mujoco.mj_step(model, data)

        # Optional: slow down real-time to see clearly
        time.sleep(0.01)
