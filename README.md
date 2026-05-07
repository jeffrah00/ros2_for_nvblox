# ros2_depth_for_nvblox

Two ROS 2 launch files that drive [Isaac ROS nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox):

- **`realsense_example.launch.py`** — depth comes from a RealSense camera's onboard stereo module (default Isaac ROS nvblox example).
- **`s2m2_example.launch.py`** — depth comes from the [S2M2](https://github.com/junhong-3dv/s2m2) stereo-matching network running on a stereo image pair. Defaults are wired to a RealSense D435i's two IR cameras (`/camera0/infra1`, `/camera0/infra2`); the topic names are launch args so any other stereo source works too. Everything downstream (vSLAM, nvblox, visualization) is identical.

## Prerequisites

- ROS 2 + Isaac ROS nvblox bringup: `nvblox_examples_bringup`, `isaac_ros_launch_utils`, `nvblox_ros_python_utils`.
- CUDA-enabled GPU + PyTorch (matching the CUDA version on your machine).
- Python deps: `cv_bridge`, `message_filters` (already pulled in by Isaac ROS), `opencv-python`, `numpy`.
- For the TensorRT path: `tensorrt`, `pycuda`.

## Install S2M2

```bash
git clone https://github.com/junhong-3dv/s2m2 && cd s2m2
pip install -e .
# download pretrained weights from upstream HuggingFace into ./weights/<S|M|L|XL>/
```

The launch file imports S2M2 lazily, so a normal `pip install -e .` is enough — no further wiring needed.

## (Optional) Export a TensorRT engine

```bash
python s2m2/export_tensorrt.py \
    --model_type S --weights_dir ./weights/S \
    --height 384 --width 640 \
    --output s2m2_S_384x640.engine
```

`height` and `width` must be divisible by 32 and must match the values you pass to the launch file.

## Run

Both backends require inference H,W to be divisible by 32. If you don't set `s2m2_width`/`s2m2_height`, the node center-crops each frame to the nearest multiple of 32 and zero-pads the resulting depth back to the original dimensions before publishing — so nvblox always sees depth aligned to the unmodified camera intrinsics.

```bash
# PyTorch (default S model, RealSense D435i IR pair):
ros2 launch s2m2_example.launch.py \
    s2m2_weights_path:=/abs/path/to/s2m2/weights/S

# TensorRT engine (height/width must match what the engine was exported with):
ros2 launch s2m2_example.launch.py \
    s2m2_engine_path:=/abs/path/to/s2m2_S_384x640.engine \
    s2m2_height:=384 s2m2_width:=640

# Custom stereo source (override topic names):
ros2 launch s2m2_example.launch.py \
    s2m2_weights_path:=/abs/path/to/weights/S \
    s2m2_left_topic:=/my_cam/left/image_rect \
    s2m2_right_topic:=/my_cam/right/image_rect \
    s2m2_camera_info_topic:=/my_cam/left/camera_info \
    s2m2_baseline_m:=0.12

# Replay from a stereo rosbag:
ros2 launch s2m2_example.launch.py \
    rosbag:=/path/to/bag s2m2_weights_path:=/abs/path/to/weights/S

# Fall back to the default RealSense onboard depth:
ros2 launch s2m2_example.launch.py depth_source:=realsense
```

The S2M2 node subscribes (defaults are RealSense D435i IR; override via the args below for any other stereo source):

| Direction | Topic (default)                                   | Notes |
| --------- | ------------------------------------------------- | ----- |
| in        | `/camera0/infra1/image_rect_raw`                  | left IR (mono8 OK; auto-converted to 3ch) |
| in        | `/camera0/infra2/image_rect_raw`                  | right IR |
| in        | `/camera0/infra1/camera_info`                     | left CameraInfo |
| out       | `/camera0/depth/image_rect_raw` (`32FC1`, meters) | nvblox subscribes here |
| out       | `/camera0/depth/camera_info`                      |       |

For the D435i specifically, the stereo baseline is ~50 mm (`s2m2_baseline_m:=0.05`, which is the default). Verify against your unit's calibration.

## Important launch args

| Arg | Default | Notes |
| --- | --- | --- |
| `depth_source` | `s2m2` | `s2m2` or `realsense` |
| `s2m2_model_type` | `S` | `S`, `M`, `L`, `XL` |
| `s2m2_num_refine` | `1` | refinement iterations |
| `s2m2_weights_path` | _empty_ | PyTorch weights dir |
| `s2m2_engine_path` | _empty_ | TensorRT `.engine` (takes priority over weights) |
| `s2m2_left_topic` | `/camera0/infra1/image_rect_raw` | RealSense D435i IR1 by default |
| `s2m2_right_topic` | `/camera0/infra2/image_rect_raw` | RealSense D435i IR2 by default |
| `s2m2_camera_info_topic` | `/camera0/infra1/camera_info` | left CameraInfo |
| `s2m2_output_depth_topic` | `/camera0/depth/image_rect_raw` | nvblox subscribes here |
| `s2m2_output_camera_info_topic` | `/camera0/depth/camera_info` | |
| `s2m2_width` | `0` | inference width; `0` = crop to nearest /32. Must be divisible by 32. |
| `s2m2_height` | `0` | inference height; same rule. |
| `s2m2_baseline_m` | `0.05` | stereo baseline in meters (left CameraInfo doesn't carry it) |
| `s2m2_confidence_threshold` | `0.0` | mask depth where conf < threshold; `0` disables |
| `s2m2_device` | `cuda` | `cuda` or `cpu` |

## Troubleshooting

- **All-zero or wildly wrong depth** — set `s2m2_baseline_m` to your actual stereo baseline in meters. The left CameraInfo's projection matrix typically does not encode it.
- **Shape error from S2M2** — make `s2m2_width`/`s2m2_height` divisible by 32, or leave them as `0` and let the node auto-crop.
- **TensorRT shape mismatch** — re-export the engine at the exact `s2m2_height`/`s2m2_width` you launch with.
- **nvblox reports no depth** — confirm `/camera0/depth/image_rect_raw` matches your nvblox version's expected depth topic; if not, override `s2m2_output_depth_topic` (and `s2m2_output_camera_info_topic`) accordingly.
- **Sync drops** — loosen the `ApproximateTimeSynchronizer` slop in `s2m2_depth_node.py` or align timestamps in your bag.
