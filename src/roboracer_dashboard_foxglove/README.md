# Follow The Gap Debug Dashboard (Python)

Video: *(coming soon)*

Use this package to see your Python Follow The Gap node in Foxglove: the gap you chose, the beam you aim at, and the safety bubble. Add three lines to your Python gap_follow node. C++ is not supported on this branch.

## 1. Add this package to your workspace

**Clone this repository into the `src` folder of your ROS 2 workspace, then build:**
```bash
cd src
git clone -b ftg-python https://github.com/aweSamePage/Dashboard_Foxglove.git
```

**Check**
```bash
cd Dashboard_Foxglove && git branch --show-current
```
If "ftg-python" shows up you are good to go.

```bash
cd <your_ws> #above your original src directory
colcon build --packages-select roboracer_dashboard_foxglove
source install/setup.bash
```

## 2. Import the Foxglove layout

File: `src/roboracer_dashboard_foxglove/foxglove/layout_ftg_debug.json`

In Foxglove:

1. Open **Layouts** (left sidebar)
2. Click **+ Add** → **Import personal layout**
3. Select `layout_ftg_debug.json`
4. Click **Open**

## 3. Add three lines to your gap_follow node

The names on the right are examples. Use your own variables.

```python
from roboracer_dashboard_foxglove.controller_debug import FtgDebugPublisher

# __init__:
self._debug = FtgDebugPublisher(self)

# after you publish /drive: (this is an example)
self._debug.publish(
    scan=data,                 # LaserScan from your lidar callback
    ranges=proc_ranges,        # the array you actually use for gap / AIM / bubble
    steer=steering_command,    # steering you publish to /drive, in radians
    speed=velocity_command,    # speed you publish to /drive
    gap=gap,                   # (start, end) beam indices, or (None, None)
    best_point=best_point,     # AIM beam index in that same ranges array; None if no gap
)
```

## 4. Build and run

```bash
cd <your_ws>
colcon build
source install/setup.bash
```

1. Start the simulator:
   ```bash
   ros2 launch f1tenth_gym_ros gym_bridge_launch.py
   ```
2. Start the Foxglove bridge:
   ```bash
   ros2 run foxglove_bridge foxglove_bridge
   ```
3. Run your gap_follow node.
4. Open Foxglove and connect to the bridge.

## 5. What you should see in 3D

- **Green wedge** — the free gap you chose
- **Yellow AIM ball** — the beam you are steering toward
- **Red BUBBLE** — safety bubble around the closest obstacle

| What you see in 3D | Advice | What to change |
|---|---|---|
| Yellow AIM jumps left/right on a straight | `[Straight wobble]` | Aim at the gap midpoint |
| Yellow AIM sits off-center in a gap (not jumping) | `[Far AIM]` | Aim at the gap midpoint, not the farthest beam |
| Yellow AIM sits on the edge of the green gap | `[Corner AIM]` | Use the gap midpoint |
| Steering is large, speed is still high, and you are about to hit a wall | `[Corner speed]` | Scale speed down when steering is large |
| Yellow AIM goes one way, the car steers the other | `[Steer sign]` | Left is positive. Flip the sign of the steering angle |
| Red BUBBLE is tiny and you get close to a wall on a straight | `[Bubble too small]` | Increase the safety bubble |
| Red BUBBLE ate the gap / AIM on a wall | `[Bubble too large]` | Shrink the safety bubble |
| Green gap / yellow AIM sit at the wrong angle | `[Chunking]` | Pass the processed lidar, not the raw slice |
| The lidar window includes beams behind the car | `[Rear scan]` | Use only the forward slice |
| Steering is huge (degrees or a beam index) | `[Steer units]` | Use a radian steering angle |
| Yellow AIM is off to the side, steering is ~0 | `[Steer unused]` | Convert the AIM beam to a steering angle in radians |

---

## Coming later (WIP)

Pure Pursuit, MPC, and MPPI dashboards are still in progress. Do not use them yet.
This is more of a basic level advice so it is based on simulator. When you are pushing to the limit with real cars, please do not listen to these advices. 
