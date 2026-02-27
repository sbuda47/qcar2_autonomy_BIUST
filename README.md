# qcar2_autonomy_BIUST

Modular autonomous driving stack for the Quanser QCar 2 (Isaac ROS container workflow).
This project restructures the working “monolithic” driver into a clean, maintainable
package layout (compat, control, perception, planning, platform), while keeping the
runtime behavior consistent.

## What this repo contains

- `qcar2_autonomy/app/driver_main.py`  
  Main entrypoint used to run the stack.

- `qcar2_autonomy/compat/`  
  Compatibility patches used in container / virtual environments (e.g., `getlogin` patch
  and virtual QCar config patch).

- `qcar2_autonomy/control/`  
  Control components (speed controller, steering controller, limits).

- `qcar2_autonomy/perception/`  
  Lane perception + optional YOLO road-sign rules pipeline (if a valid model is provided).

- `scripts/run_competition_stack.sh`  
  Convenience launcher (optional).

> Note: YOLO weights (`road_signs.pt`) are NOT included in this repo. If your
> `road_signs.pt` is empty or missing, run in lane-follow-only mode (no YOLO).

---

## Environment

This repo is intended to be used inside the Isaac ROS container workspace:

- Workspace root: `/workspaces/isaac_ros-dev`
- Package path: `/workspaces/isaac_ros-dev/src/qcar2_autonomy`

---

## How to run (direct Python module run)

From the terminal inside the container:

```bash
cd /workspaces/isaac_ros-dev/src/qcar2_autonomy
python3 -m qcar2_autonomy.app.driver_main
