# ros2_depth_for_nvblox

Three ROS 2 launch files that drive [Isaac ROS nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox):

- **`launch/realsense_example.launch.py`** — depth comes from a RealSense camera's onboard stereo module (default Isaac ROS nvblox example).
- **`launch/s2m2_example.launch.py`** — depth comes from the [S2M2](https://github.com/junhong-3dv/s2m2) stereo-matching network running on a stereo image pair. Defaults are wired to a RealSense D435i's two IR cameras (`/camera0/infra1`, `/camera0/infra2`); the topic names are launch args so any other stereo source works too. Everything downstream (vSLAM, nvblox, visualization) is identical.
- **`launch/custom_depth_example.launch.py`** — model-agnostic version of the S2M2 example. Loads any stereo network from an **ONNX `.onnx`** file (via `onnxruntime-gpu`) or a pre-built **TensorRT `.engine`**. Use this when you have pretrained weights from any other repo: export them to ONNX once and you're done.

```
ros2_depth_for_nvblox/
├── launch/        # ros2 launch entry points
│   ├── realsense_example.launch.py
│   ├── s2m2_example.launch.py
│   └── custom_depth_example.launch.py
└── scripts/       # rclpy nodes the launch files spawn
    ├── s2m2_depth_node.py
    └── custom_depth_node.py
```

## Prerequisites

- ROS 2 + Isaac ROS nvblox bringup: `nvblox_examples_bringup`, `isaac_ros_launch_utils`, `nvblox_ros_python_utils`.
- CUDA-enabled GPU + PyTorch (matching the CUDA version on your machine).
- Python deps: `cv_bridge`, `message_filters` (already pulled in by Isaac ROS), `opencv-python`, `numpy`.
- For the TensorRT path: `tensorrt` + `pycuda`. Both TRT 10.x (tensor-name API) and TRT ≤9.x (binding-index API) are supported — the node detects which is available and uses the matching path.
- For the generic `custom_depth_example.launch.py` ONNX path: `onnxruntime-gpu` (`pip install onnxruntime-gpu`).

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
ros2 launch launch/s2m2_example.launch.py \
    s2m2_weights_path:=/abs/path/to/s2m2/weights/S

# TensorRT engine (height/width must match what the engine was exported with):
ros2 launch launch/s2m2_example.launch.py \
    s2m2_engine_path:=/abs/path/to/s2m2_S_384x640.engine \
    s2m2_height:=384 s2m2_width:=640

# Custom stereo source (override topic names):
ros2 launch launch/s2m2_example.launch.py \
    s2m2_weights_path:=/abs/path/to/weights/S \
    s2m2_left_topic:=/my_cam/left/image_rect \
    s2m2_right_topic:=/my_cam/right/image_rect \
    s2m2_camera_info_topic:=/my_cam/left/camera_info \
    s2m2_baseline_m:=0.12

# Replay from a stereo rosbag:
ros2 launch launch/s2m2_example.launch.py \
    rosbag:=/path/to/bag s2m2_weights_path:=/abs/path/to/weights/S

# Fall back to the default RealSense onboard depth:
ros2 launch launch/s2m2_example.launch.py depth_source:=realsense
```

The S2M2 node subscribes (defaults are RealSense D435i IR; override via the args below for any other stereo source):

| Direction | Topic (default)                                   | Notes |
| --------- | ------------------------------------------------- | ----- |
| in        | `/camera0/infra1/image_rect_raw`                  | left IR (mono8 OK; auto-converted to 3ch) |
| in        | `/camera0/infra2/image_rect_raw`                  | right IR |
| in        | `/camera0/infra1/camera_info`                     | left CameraInfo |
| out       | `/s2m2/depth/image_rect_raw` (`32FC1`, meters)    | distinguishable from RealSense splitter's `/camera0/depth/...`; nvblox is remapped to it |
| out       | `/s2m2/depth/camera_info`                         |       |

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
| `s2m2_output_depth_topic` | `/s2m2/depth/image_rect_raw` | nvblox is remapped to this |
| `s2m2_output_camera_info_topic` | `/s2m2/depth/camera_info` | |
| `s2m2_width` | `0` | inference width; `0` = crop to nearest /32. Must be divisible by 32. |
| `s2m2_height` | `0` | inference height; same rule. |
| `s2m2_baseline_m` | `0.05` | stereo baseline in meters (left CameraInfo doesn't carry it) |
| `s2m2_confidence_threshold` | `0.0` | mask depth where conf < threshold; `0` disables |
| `s2m2_device` | `cuda` | `cuda` or `cpu` |
| `rviz_config` | _empty_ | optional path to a `.rviz` file; forwarded to upstream visualization.launch.py |

## Generic example (`custom_depth_example.launch.py`)

Same pipeline as the S2M2 example, but the depth node loads an ONNX or TensorRT export of *any* stereo-matching model. Most upstream repos (S2M2 included) ship an ONNX export script — export your pretrained weights once and run.

```bash
# Export your model to ONNX (S2M2 shown; other repos have similar scripts):
python s2m2/export_onnx.py \
    --model_type S --weights_dir ./weights/S \
    --height 384 --width 640 \
    --output stereo_S_384x640.onnx

# Run nvblox with the ONNX model:
ros2 launch launch/custom_depth_example.launch.py \
    custom_onnx_path:=/abs/path/to/stereo_S_384x640.onnx \
    custom_height:=384 custom_width:=640 \
    custom_baseline_m:=0.05

# Or with a TensorRT engine (faster):
ros2 launch launch/custom_depth_example.launch.py \
    custom_engine_path:=/abs/path/to/stereo_S_384x640.engine \
    custom_height:=384 custom_width:=640
```

The generic node assumes the model takes **two inputs** `(1, 3, H, W)` (left, right; float32 in `[0,1]` by default — set `custom_input_scale:=1.0` for raw uint8) and returns **1, 2, or 3 outputs** in order: disparity `(1, H, W)`, optional occlusion mask, optional confidence map. If your model has different conventions, re-export to match.

Defaults publish to `/custom_depth/depth/image_rect_raw` and `/custom_depth/depth/camera_info`; nvblox is remapped to those via the launch file's `SetRemap`. All other launch args mirror the S2M2 example with a `custom_` prefix instead of `s2m2_`.

## Troubleshooting

- **All-zero or wildly wrong depth** — set `s2m2_baseline_m` to your actual stereo baseline in meters. The left CameraInfo's projection matrix typically does not encode it.
- **Shape error from S2M2** — make `s2m2_width`/`s2m2_height` divisible by 32, or leave them as `0` and let the node auto-crop.
- **TensorRT shape mismatch** — re-export the engine at the exact `s2m2_height`/`s2m2_width` you launch with.
- **nvblox reports no depth** — confirm the launch file's `SetRemap` source (`/camera0/depth/image_rect_raw`) matches your nvblox version's expected depth topic. The S2M2 node publishes to `/s2m2/depth/...` and the remap routes nvblox to it; override `s2m2_output_depth_topic` / `s2m2_output_camera_info_topic` to publish under a different name.
- **Sync drops** — loosen the `ApproximateTimeSynchronizer` slop in `scripts/s2m2_depth_node.py` (or `scripts/custom_depth_node.py`) or align timestamps in your bag.
