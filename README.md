# ros2_depth_for_nvblox

Two ROS 2 launch files that drive [Isaac ROS nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox):

- **`realsense_example.launch.py`** — depth comes from a RealSense camera's onboard stereo module (default Isaac ROS nvblox example).
- **`s2m2_example.launch.py`** — depth comes from the [S2M2](https://github.com/junhong-3dv/s2m2) stereo-matching network running on a generic stereo image pair. Everything downstream (vSLAM, nvblox, visualization) is identical.

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
# PyTorch (default S model variant):
ros2 launch s2m2_example.launch.py \
    s2m2_weights_path:=/abs/path/to/s2m2/weights/S \
    s2m2_baseline_m:=0.12

# TensorRT engine (height/width must match what the engine was exported with):
ros2 launch s2m2_example.launch.py \
    s2m2_engine_path:=/abs/path/to/s2m2_S_384x640.engine \
    s2m2_height:=384 s2m2_width:=640 \
    s2m2_baseline_m:=0.12

# Replay from a stereo rosbag:
ros2 launch s2m2_example.launch.py \
    rosbag:=/path/to/bag s2m2_weights_path:=/abs/path/to/weights/S

# Fall back to the default RealSense behavior:
ros2 launch s2m2_example.launch.py depth_source:=realsense
```

The S2M2 node subscribes (default topics, override via the args below):

| Direction | Topic                          |
| --------- | ------------------------------ |
| in        | `/stereo/left/image_rect`      |
| in        | `/stereo/right/image_rect`     |
| in        | `/stereo/left/camera_info`     |
| out       | `/camera0/depth/image_rect_raw` (`32FC1`, meters) |
| out       | `/camera0/depth/camera_info`   |

## Important launch args

| Arg | Default | Notes |
| --- | --- | --- |
| `depth_source` | `s2m2` | `s2m2` or `realsense` |
| `s2m2_model_type` | `S` | `S`, `M`, `L`, `XL` |
| `s2m2_num_refine` | `1` | refinement iterations |
| `s2m2_weights_path` | _empty_ | PyTorch weights dir |
| `s2m2_engine_path` | _empty_ | TensorRT `.engine` (takes priority over weights) |
| `s2m2_left_topic` | `/stereo/left/image_rect` | |
| `s2m2_right_topic` | `/stereo/right/image_rect` | |
| `s2m2_camera_info_topic` | `/stereo/left/camera_info` | |
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
