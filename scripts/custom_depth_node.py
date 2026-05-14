#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generic stereo-depth ROS 2 node.

Subscribes to a synchronized stereo pair (left/right rectified images) plus the
left CameraInfo, runs an arbitrary stereo-matching model to produce disparity,
converts disparity to metric depth, and publishes a depth image + CameraInfo.

Two backends are supported (engine_path takes priority):
  - ONNX:     pass a `.onnx` file, loaded via onnxruntime-gpu.
  - TensorRT: pass a `.engine` built from your model's ONNX export.

Both are framework-agnostic exports, so this node does NOT need access to the
original PyTorch model class. To use a `.pth` state-dict, first export to ONNX
with the upstream repo's export script (e.g. S2M2 ships `export_onnx.py`).

Model interface assumed:
  inputs  : (left, right), each shape (1, 3, H, W), float32 in [0,1] by default
  outputs : 1, 2, or 3 tensors in order:
              disparity         (1, H, W)
              occlusion?        (1, H, W)  (>0.5 marks occluded)
              confidence?       (1, H, W)  (compared with confidence_threshold)
"""

import os
import sys
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters


def _to_3channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


class CustomStereoDepthNode(Node):
    def __init__(self):
        super().__init__('custom_depth_node')
        self._declare_params()
        self._setup_backend()
        self._setup_io()
        self.bridge = CvBridge()

    # ------------------------------------------------------------------ params
    def _declare_params(self):
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('left_topic', '/camera0/infra1/image_rect_raw')
        self.declare_parameter('right_topic', '/camera0/infra2/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera0/infra1/camera_info')
        self.declare_parameter('output_depth_topic', '/custom_depth/depth/image_rect_raw')
        self.declare_parameter('output_camera_info_topic', '/custom_depth/depth/camera_info')
        self.declare_parameter('width', 0)
        self.declare_parameter('height', 0)
        self.declare_parameter('baseline_m', 0.05)
        self.declare_parameter('confidence_threshold', 0.0)
        self.declare_parameter('input_scale', 1.0 / 255.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('fps_log_period', 5.0)
        self.declare_parameter('stall_warn_ms', 250.0)
        self.declare_parameter('use_cuda_graph', True)
        # Run the disparity->depth mask+divide on the GPU instead of CPU, so
        # only the final (H,W) depth gets D2H'd (vs disp+occ+conf D2H + CPU
        # np.divide). Saves ~0.5-1.5 ms/frame at typical resolutions. Off if
        # the engine doesn't emit float32 outputs or kernel compile fails.
        self.declare_parameter('use_gpu_depth', True)
        self.declare_parameter('use_sensor_qos', False)
        self.declare_parameter('process_every_n', 1)
        # Direct librealsense capture: skip the ROS Image subscription on the
        # input side and pull IR1/IR2 frames straight from pyrealsense2 in this
        # process. Requires no realsense2_camera_node running on the same
        # device (librealsense allows one process per device). Off by default.
        self.declare_parameter('camera_source', 'ros')         # 'ros' | 'realsense'
        self.declare_parameter('rs_serial', '')
        self.declare_parameter('rs_width', 640)
        self.declare_parameter('rs_height', 480)
        self.declare_parameter('rs_fps', 30)
        self.declare_parameter('rs_frame_id', 'camera_infra1_optical_frame')
        self.declare_parameter('rs_disable_emitter', True)

        self.cfg_width = int(self.get_parameter('width').value)
        self.cfg_height = int(self.get_parameter('height').value)
        if self.cfg_width and self.cfg_width % 32 != 0:
            self.get_logger().fatal(f'width={self.cfg_width} must be divisible by 32')
            raise SystemExit(1)
        if self.cfg_height and self.cfg_height % 32 != 0:
            self.get_logger().fatal(f'height={self.cfg_height} must be divisible by 32')
            raise SystemExit(1)
        self.baseline_m = float(self.get_parameter('baseline_m').value)
        self.conf_thr = float(self.get_parameter('confidence_threshold').value)
        self.input_scale = float(self.get_parameter('input_scale').value)
        self._fps_log_period_s = float(self.get_parameter('fps_log_period').value)
        self._stall_warn_ms = float(self.get_parameter('stall_warn_ms').value)
        self._process_every_n = max(1, int(self.get_parameter('process_every_n').value))
        self._sync_count = 0
        self._fps_window_start = time.perf_counter()
        self._fps_frame_count = 0
        self._fps_stage_acc = {}  # stage name -> accumulated ms within the window
        self._fps_stage_max = {}  # stage name -> max ms within the window
        self._logged_first_frame = False
        self._prev_callback_end = None  # perf_counter at end of previous on_stereo
        self._last_infer_timing = {}    # sub-stage ms from the last inference call
        self._incoming_count = 0        # left-image msgs received since last FPS log
        self._outgoing_count = 0        # depth msgs delivered back to us since last log
        self._depth_buf = None          # reused (H,W) float32 depth-at-inference-res
        self._depth_full_buf = None     # reused (h_orig,w_orig) float32 output (crop mode)
        self._depth_msg = None          # reused Image msg for the depth output
        self._info_out = None           # cached CameraInfo (intrinsics are static)

    # ----------------------------------------------------------------- backend
    def _setup_backend(self):
        engine = self.get_parameter('engine_path').value
        onnx_path = self.get_parameter('onnx_path').value
        device_str = self.get_parameter('device').value

        if engine and os.path.isfile(engine):
            self._init_trt(engine)
            self.backend = 'trt'
            self.get_logger().info(f'Backend: TensorRT engine ({engine})')
        elif onnx_path and os.path.isfile(onnx_path):
            self._init_onnx(onnx_path, device_str)
            self.backend = 'onnx'
            self.get_logger().info(f'Backend: ONNX Runtime ({onnx_path}, providers={self.ort_sess.get_providers()})')
        else:
            self.get_logger().fatal(
                'Need either onnx_path (.onnx) or engine_path (.engine) to be set.')
            raise SystemExit(1)

    def _init_onnx(self, onnx_path: str, device_str: str):
        import onnxruntime as ort
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device_str == 'cuda' else ['CPUExecutionProvider'])
        self.ort_sess = ort.InferenceSession(onnx_path, providers=providers)
        self.ort_input_names = [i.name for i in self.ort_sess.get_inputs()]
        self.ort_output_names = [o.name for o in self.ort_sess.get_outputs()]
        if len(self.ort_input_names) != 2:
            self.get_logger().fatal(
                f'ONNX model has {len(self.ort_input_names)} inputs; expected 2 (left, right).')
            raise SystemExit(1)
        n_out = len(self.ort_output_names)
        if n_out < 1 or n_out > 3:
            self.get_logger().fatal(
                f'ONNX model has {n_out} outputs; expected 1 (disp), 2 (disp, occ), or 3 (disp, occ, conf).')
            raise SystemExit(1)

    def _init_trt(self, engine_path: str):
        if not (self.cfg_width and self.cfg_height):
            self.get_logger().fatal(
                'TensorRT path requires width and height to be set explicitly '
                '(both divisible by 32, matching the engine).')
            raise SystemExit(1)
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()
        self._trt = trt
        self._cuda = cuda

        # TRT >=10 removed the binding-index API in favor of tensor names.
        # Detect which API to use; both paths populate self.trt_io the same way.
        self._trt_use_tensor_api = hasattr(self.trt_engine, 'num_io_tensors')
        H, W = self.cfg_height, self.cfg_width
        self.trt_io = []

        if self._trt_use_tensor_api:
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                is_input = (self.trt_engine.get_tensor_mode(name)
                            == trt.TensorIOMode.INPUT)
                shape = tuple(self.trt_engine.get_tensor_shape(name))
                if -1 in shape and is_input:
                    self.trt_context.set_input_shape(name, (1, 3, H, W))
                    shape = tuple(self.trt_context.get_tensor_shape(name))
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                host_buf = cuda.pagelocked_empty(shape, dtype)  # page-locked: fast DMA, graph-capturable
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                self.trt_context.set_tensor_address(name, int(dev_buf))
                self.trt_io.append({
                    'name': name, 'is_input': is_input, 'shape': shape,
                    'dtype': dtype, 'host': host_buf, 'dev': dev_buf,
                })
            self._trt_stream = cuda.Stream()
            self.trt_bindings = None
        else:
            self.trt_bindings = [None] * self.trt_engine.num_bindings
            for i in range(self.trt_engine.num_bindings):
                shape = tuple(self.trt_engine.get_binding_shape(i))
                is_input = self.trt_engine.binding_is_input(i)
                if -1 in shape and is_input:
                    self.trt_context.set_binding_shape(i, (1, 3, H, W))
                    shape = (1, 3, H, W)
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
                host_buf = cuda.pagelocked_empty(shape, dtype)
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                self.trt_bindings[i] = int(dev_buf)
                self.trt_io.append({
                    'name': self.trt_engine.get_binding_name(i),
                    'is_input': is_input, 'shape': shape, 'dtype': dtype,
                    'host': host_buf, 'dev': dev_buf,
                })
            self._trt_stream = None

        n_in = sum(1 for io in self.trt_io if io['is_input'])
        n_out = sum(1 for io in self.trt_io if not io['is_input'])
        if n_in != 2:
            self.get_logger().fatal(f'Engine has {n_in} inputs; expected 2 (left, right).')
            raise SystemExit(1)
        if n_out < 1 or n_out > 3:
            self.get_logger().fatal(
                f'Engine has {n_out} outputs; expected 1 (disp), 2 (disp, occ), or 3 (disp, occ, conf).')
            raise SystemExit(1)

        # CUDA-graph capture of H2D->execute->(optional depth kernel)->D2H.
        # Valid because the page-locked host buffers and device buffers above
        # have fixed addresses for the node's lifetime; only their contents change.
        self._use_cuda_graph = (bool(self.get_parameter('use_cuda_graph').value)
                                and self._trt_use_tensor_api)
        self._trt_graph_exec = None
        self._trt_warmup_left = 2  # TRT needs a couple of real runs before capture

        # GPU disparity->depth kernel. Replaces the CPU mask + np.divide path
        # and the D2H of disp/occ/conf with a single D2H of (H,W) depth.
        # custom_depth_node: occ polarity is *inverted* relative to s2m2
        # (occ > 0.5 means occluded here), and conf_thr is variable.
        self._use_gpu_depth = bool(self.get_parameter('use_gpu_depth').value)
        self._depth_kernel = None
        self._depth_dev = None
        self._depth_host = None
        self._depth_fx_baseline_captured = None
        if self._use_gpu_depth:
            out_ios = [io for io in self.trt_io if not io['is_input']]
            if any(io['dtype'] != np.float32 for io in out_ios):
                self.get_logger().warn(
                    'use_gpu_depth requires float32 engine outputs. '
                    'Falling back to CPU mask path.')
                self._use_gpu_depth = False
            else:
                self._n_out = len(out_ios)
                disp_shape = out_ios[0]['shape']
                H_out, W_out = disp_shape[-2], disp_shape[-1]
                self._n_pixels = int(H_out * W_out)
                self._depth_host = cuda.pagelocked_empty((H_out, W_out), np.float32)
                self._depth_dev = cuda.mem_alloc(self._depth_host.nbytes)
                # For absent outputs (engine emits fewer than 3), reuse the
                # disp device pointer as a stand-in -- the kernel won't read
                # it because has_occ / has_conf will be 0.
                self._occ_dev_arg = out_ios[1]['dev'] if self._n_out >= 2 else out_ios[0]['dev']
                self._conf_dev_arg = out_ios[2]['dev'] if self._n_out >= 3 else out_ios[0]['dev']
                self._has_occ = 1 if self._n_out >= 2 else 0
                # conf_thr <= 0 means "don't filter on confidence" in the CPU path.
                self._has_conf_filter = 1 if (self._n_out >= 3 and self.conf_thr > 0.0) else 0
                try:
                    from pycuda.compiler import SourceModule
                    _src = r"""
                    extern "C" __global__
                    void disp_to_depth(const float *disp, const float *occ,
                                       const float *conf, float *depth,
                                       float fx_baseline, float eps, float conf_thr,
                                       int has_occ, int has_conf, int n) {
                        int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i >= n) return;
                        float d = disp[i];
                        bool valid = d > eps;
                        if (has_occ)  valid = valid && (occ[i]  <= 0.5f);
                        if (has_conf) valid = valid && (conf[i] >= conf_thr);
                        depth[i] = valid ? (fx_baseline / d) : 0.0f;
                    }
                    """
                    self._depth_mod = SourceModule(_src, no_extern_c=True)
                    self._depth_kernel = self._depth_mod.get_function('disp_to_depth')
                    self._depth_block = (256, 1, 1)
                    self._depth_grid = ((self._n_pixels + 255) // 256, 1, 1)
                    self.get_logger().info(
                        f'GPU disp->depth kernel ready (n_pixels={self._n_pixels}, '
                        f'n_out={self._n_out}, has_occ={self._has_occ}, '
                        f'conf_thr={self.conf_thr}).')
                except Exception as e:
                    self.get_logger().warn(
                        f'GPU depth kernel compile failed ({e}); falling back to CPU mask path.')
                    self._use_gpu_depth = False
                    self._depth_kernel = None

    # --------------------------------------------------------------------- io
    def _setup_io(self):
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        out_depth = self.get_parameter('output_depth_topic').value
        out_info = self.get_parameter('output_camera_info_topic').value
        self.camera_source = self.get_parameter('camera_source').value
        if self.camera_source not in ('ros', 'realsense'):
            self.get_logger().fatal(
                f'camera_source must be "ros" or "realsense"; got "{self.camera_source}".')
            raise SystemExit(1)

        # Optional sensor-data QoS (BEST_EFFORT, KEEP_LAST 5). Off by default: a
        # RELIABLE subscriber (some nvblox configs) is incompatible with a
        # BEST_EFFORT publisher, which would silently stop depth from flowing.
        if bool(self.get_parameter('use_sensor_qos').value):
            from rclpy.qos import qos_profile_sensor_data
            sub_kw = {'qos_profile': qos_profile_sensor_data}
            pub_qos = qos_profile_sensor_data
            plain_qos = qos_profile_sensor_data
        else:
            sub_kw = {}
            pub_qos = 5
            plain_qos = 10

        self.pub_depth = self.create_publisher(Image, out_depth, pub_qos)
        self.pub_info = self.create_publisher(CameraInfo, out_info, pub_qos)

        if self.camera_source == 'ros':
            left_sub = message_filters.Subscriber(self, Image, left_topic, **sub_kw)
            right_sub = message_filters.Subscriber(self, Image, right_topic, **sub_kw)
            info_sub = message_filters.Subscriber(self, CameraInfo, info_topic, **sub_kw)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [left_sub, right_sub, info_sub], queue_size=10, slop=0.05)
            self.sync.registerCallback(self.on_stereo)
            self._incoming_sub = self.create_subscription(
                Image, left_topic, self._incoming_cb, plain_qos)
            self._outgoing_sub = self.create_subscription(
                Image, out_depth, self._outgoing_cb, plain_qos)
            self.get_logger().info(
                f'subscribed: {left_topic}, {right_topic}, {info_topic}\n'
                f'publishing: {out_depth}, {out_info}')
        else:
            self._init_direct_camera()
            self._outgoing_sub = self.create_subscription(
                Image, out_depth, self._outgoing_cb, plain_qos)
            self.get_logger().info(
                f'direct librealsense capture; publishing: {out_depth}, {out_info}')

    def _init_direct_camera(self):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            self.get_logger().fatal(
                f'camera_source=realsense requires pyrealsense2 (`pip install pyrealsense2`): {e}')
            raise SystemExit(1)
        rs_width = int(self.get_parameter('rs_width').value)
        rs_height = int(self.get_parameter('rs_height').value)
        rs_fps = int(self.get_parameter('rs_fps').value)
        rs_serial = self.get_parameter('rs_serial').value
        self._rs_frame_id = self.get_parameter('rs_frame_id').value
        disable_emitter = bool(self.get_parameter('rs_disable_emitter').value)

        cfg = rs.config()
        if rs_serial:
            cfg.enable_device(rs_serial)
        cfg.enable_stream(rs.stream.infrared, 1, rs_width, rs_height, rs.format.y8, rs_fps)
        cfg.enable_stream(rs.stream.infrared, 2, rs_width, rs_height, rs.format.y8, rs_fps)
        self._rs_pipeline = rs.pipeline()
        profile = self._rs_pipeline.start(cfg)
        if disable_emitter:
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 0)

        ir1_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        intr = ir1_profile.get_intrinsics()
        info = CameraInfo()
        info.height = intr.height
        info.width = intr.width
        info.distortion_model = 'plumb_bob'
        info.d = list(intr.coeffs)
        info.k = [intr.fx, 0.0, intr.ppx,
                  0.0, intr.fy, intr.ppy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [intr.fx, 0.0, intr.ppx, 0.0,
                  0.0, intr.fy, intr.ppy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self._rs_info = info
        self.get_logger().info(
            f'librealsense started: {intr.width}x{intr.height}@{rs_fps} Hz, '
            f'fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f} '
            f'emitter={"off" if disable_emitter else "on"}.')
        self._direct_timer = self.create_timer(
            1.0 / max(rs_fps * 4, 60), self._direct_capture_tick)

    def _direct_capture_tick(self):
        success, frames = self._rs_pipeline.try_wait_for_frames(timeout_ms=0)
        if not success:
            return
        left_f = frames.get_infrared_frame(1)
        right_f = frames.get_infrared_frame(2)
        if not left_f or not right_f:
            return
        self._sync_count += 1
        if self._process_every_n > 1 and (self._sync_count % self._process_every_n):
            return
        t_start = time.perf_counter()
        left = np.asarray(left_f.get_data()).copy()
        right = np.asarray(right_f.get_data()).copy()
        left = _to_3channel(left)
        right = _to_3channel(right)
        t_decode = time.perf_counter()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._rs_frame_id
        self._process_stereo(left, right, self._rs_info, header, t_start, t_decode)

    def _incoming_cb(self, _msg: Image):
        self._incoming_count += 1

    def _outgoing_cb(self, _msg: Image):
        self._outgoing_count += 1

    # --------------------------------------------------------------- callback
    def on_stereo(self, left_msg: Image, right_msg: Image, info_msg: CameraInfo):
        self._sync_count += 1
        if self._process_every_n > 1 and (self._sync_count % self._process_every_n):
            return
        t_start = time.perf_counter()
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
        left = _to_3channel(left)
        right = _to_3channel(right)
        if left.dtype != np.uint8:
            left = left.astype(np.uint8)
            right = right.astype(np.uint8)
        t_decode = time.perf_counter()
        self._process_stereo(left, right, info_msg, left_msg.header, t_start, t_decode)

    def _process_stereo(self, left, right, info_msg, header, t_start, t_decode):
        """Pre/inference/post/publish for an already-decoded HxWx3 uint8 pair.

        Shared between the ROS message_filters path (on_stereo) and the direct
        librealsense capture path (_direct_capture_tick).
        """
        h_orig, w_orig = left.shape[:2]
        if self.cfg_width and self.cfg_height:
            W, H = self.cfg_width, self.cfg_height
            sx, sy = W / w_orig, H / h_orig
            ox = oy = 0
            left_in = cv2.resize(left, (W, H), interpolation=cv2.INTER_LINEAR)
            right_in = cv2.resize(right, (W, H), interpolation=cv2.INTER_LINEAR)
            mode = 'resize'
        else:
            H = (h_orig // 32) * 32
            W = (w_orig // 32) * 32
            if H == 0 or W == 0:
                self.get_logger().warn(
                    f'Input {h_orig}x{w_orig} is smaller than 32 in some dim; skipping frame.')
                return
            sx = sy = 1.0
            ox = (w_orig - W) // 2
            oy = (h_orig - H) // 2
            left_in = left[oy:oy + H, ox:ox + W]
            right_in = right[oy:oy + H, ox:ox + W]
            mode = 'crop'

        # NCHW float32 with input_scale (default 1/255 -> [0,1]).
        left_np = (left_in.transpose(2, 0, 1)[None].astype(np.float32) * self.input_scale)
        right_np = (right_in.transpose(2, 0, 1)[None].astype(np.float32) * self.input_scale)
        t_pre = time.perf_counter()

        # Inference. Each branch fills self._last_infer_timing with sub-stage ms.
        # TRT path with use_gpu_depth=True returns depth directly (occ/conf are
        # consumed inside the GPU kernel and never D2H'd). ONNX path and TRT
        # fallback path still return (disp, occ, conf).
        gpu_depth_used = (self.backend == 'trt' and self._use_gpu_depth)
        if self.backend == 'onnx':
            disp, occ, conf = self._onnx_infer(left_np, right_np)
        else:
            fx_baseline = float(info_msg.k[0]) * sx * self.baseline_m
            disp, occ, conf = self._trt_infer(left_np, right_np, fx_baseline)
        t_infer = time.perf_counter()

        if gpu_depth_used:
            # GPU kernel already produced depth in self._depth_host.
            depth = disp
        else:
            # Disparity -> depth at the inference resolution. One pass: build the
            # valid mask (incl. optional occ/conf) once, then divide into a pre-zeroed
            # buffer. occ is masked-out where occ > 0.5; conf where conf < conf_thr.
            fx_used = float(info_msg.k[0]) * sx
            valid = disp > 1e-3
            if occ is not None:
                valid &= (occ <= 0.5)
            if conf is not None and self.conf_thr > 0.0:
                valid &= (conf >= self.conf_thr)
            if self._depth_buf is None or self._depth_buf.shape != disp.shape:
                self._depth_buf = np.zeros(disp.shape, dtype=np.float32)
            depth = self._depth_buf
            depth.fill(0.0)
            np.divide(fx_used * self.baseline_m, disp, out=depth, where=valid)

        # Restore to original dimensions.
        if mode == 'resize':
            depth_full = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            if self._depth_full_buf is None or self._depth_full_buf.shape != (h_orig, w_orig):
                self._depth_full_buf = np.zeros((h_orig, w_orig), dtype=np.float32)
            depth_full = self._depth_full_buf
            depth_full[oy:oy + H, ox:ox + W] = depth
        t_post = time.perf_counter()

        # Build the depth Image message by hand (skips cv_bridge's per-call
        # encoding lookups / wrapper). Same bytes on the wire: cv2_to_imgmsg also
        # just does arr.tobytes(). depth_full is C-contiguous in both modes.
        if self._depth_msg is None:
            self._depth_msg = Image()
            self._depth_msg.encoding = '32FC1'
            self._depth_msg.is_bigendian = 0
        depth_msg = self._depth_msg
        depth_msg.height = depth_full.shape[0]
        depth_msg.width = depth_full.shape[1]
        depth_msg.step = depth_full.shape[1] * 4
        depth_msg.header = header
        depth_msg.data = depth_full.tobytes()
        self.pub_depth.publish(depth_msg)

        # CameraInfo intrinsics are static per camera: cache and reuse, refreshing
        # only if k changes; each frame just stamp the header.
        if self._info_out is None or list(self._info_out.k) != list(info_msg.k):
            info_out = CameraInfo()
            info_out.height = info_msg.height
            info_out.width = info_msg.width
            info_out.distortion_model = info_msg.distortion_model
            info_out.d = list(info_msg.d)
            info_out.k = list(info_msg.k)
            info_out.r = list(info_msg.r)
            info_out.p = list(info_msg.p)
            info_out.binning_x = info_msg.binning_x
            info_out.binning_y = info_msg.binning_y
            info_out.roi = info_msg.roi
            self._info_out = info_out
        self._info_out.header = header
        self.pub_info.publish(self._info_out)
        t_pub = time.perf_counter()

        if not self._logged_first_frame:
            self._logged_first_frame = True
            self.get_logger().info(
                f'first frame: input {w_orig}x{h_orig} -> inference {W}x{H} ({mode}); '
                f'backend={self.backend}')

        idle_ms = ((t_start - self._prev_callback_end) * 1e3
                   if self._prev_callback_end is not None else 0.0)
        self._prev_callback_end = t_pub
        stages = {
            'idle': idle_ms,
            'decode': (t_decode - t_start) * 1e3,
            'pre': (t_pre - t_decode) * 1e3,
        }
        stages.update(self._last_infer_timing)
        stages['post'] = (t_post - t_infer) * 1e3
        stages['pub'] = (t_pub - t_post) * 1e3
        if self._stall_warn_ms > 0.0:
            big = {k: v for k, v in stages.items() if v >= self._stall_warn_ms}
            if big:
                self.get_logger().warn(
                    'long gap: ' + ', '.join(f'{k}={v:.0f}ms' for k, v in big.items()))
        self._update_fps(stages)

    # ----------------------------------------------------------------- fps log
    def _update_fps(self, stage_ms: dict):
        if self._fps_log_period_s <= 0.0:
            return
        self._fps_frame_count += 1
        for name, ms in stage_ms.items():
            self._fps_stage_acc[name] = self._fps_stage_acc.get(name, 0.0) + ms
            self._fps_stage_max[name] = max(self._fps_stage_max.get(name, 0.0), ms)
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed >= self._fps_log_period_s:
            count = self._fps_frame_count
            breakdown = ' | '.join(
                f'{name} {self._fps_stage_acc[name] / count:.1f}/{self._fps_stage_max[name]:.1f}'
                for name in stage_ms)
            self.get_logger().info(
                f'FPS: {count / elapsed:.1f} (in {self._incoming_count / elapsed:.1f} Hz, '
                f'out {self._outgoing_count / elapsed:.1f} Hz, {count} frames in {elapsed:.1f}s) | '
                f'{breakdown} ms (avg/max)')
            self._fps_window_start = time.perf_counter()
            self._fps_frame_count = 0
            self._fps_stage_acc = {}
            self._fps_stage_max = {}
            self._incoming_count = 0
            self._outgoing_count = 0

    # ---------------------------------------------------------------- onnx fn
    def _onnx_infer(self, left_np: np.ndarray, right_np: np.ndarray):
        t0 = time.perf_counter()
        outs = self.ort_sess.run(
            self.ort_output_names,
            {self.ort_input_names[0]: left_np,
             self.ort_input_names[1]: right_np})
        t_exec = time.perf_counter()
        disp = np.squeeze(outs[0]).astype(np.float32)
        occ = np.squeeze(outs[1]).astype(np.float32) if len(outs) > 1 else None
        conf = np.squeeze(outs[2]).astype(np.float32) if len(outs) > 2 else None
        t_out = time.perf_counter()
        self._last_infer_timing = {
            'exec': (t_exec - t0) * 1e3, 'out': (t_out - t_exec) * 1e3,
        }
        return disp, occ, conf

    # ----------------------------------------------------------------- trt fn
    def _launch_depth_kernel(self, out_buffers, fx_baseline, stream):
        """Launch disp->depth kernel reading TRT output device buffers."""
        self._depth_kernel(
            out_buffers[0]['dev'], self._occ_dev_arg, self._conf_dev_arg,
            self._depth_dev,
            np.float32(fx_baseline), np.float32(1e-3), np.float32(self.conf_thr),
            np.int32(self._has_occ), np.int32(self._has_conf_filter),
            np.int32(self._n_pixels),
            block=self._depth_block, grid=self._depth_grid, stream=stream)

    def _enqueue_trt_async(self, in_buffers, out_buffers, fx_baseline=0.0):
        """H2D copies -> execute -> [depth kernel] -> D2H, on self._trt_stream."""
        stream = self._trt_stream
        for io in in_buffers:
            self._cuda.memcpy_htod_async(io['dev'], io['host'], stream)
        self.trt_context.execute_async_v3(stream.handle)
        if self._use_gpu_depth:
            self._launch_depth_kernel(out_buffers, fx_baseline, stream)
            self._cuda.memcpy_dtoh_async(self._depth_host, self._depth_dev, stream)
        else:
            for io in out_buffers:
                self._cuda.memcpy_dtoh_async(io['host'], io['dev'], stream)

    def _trt_infer(self, left_np: np.ndarray, right_np: np.ndarray,
                   fx_baseline: float = 0.0):
        in_buffers = [io for io in self.trt_io if io['is_input']]
        out_buffers = [io for io in self.trt_io if not io['is_input']]
        t0 = time.perf_counter()

        # np.copyto casts left_np/right_np (already (1,3,H,W) float32) into the
        # (page-locked) host buffer in one pass -- same result as the previous
        # astype/reshape/copyto, fewer temporaries.
        np.copyto(in_buffers[0]['host'], left_np.reshape(in_buffers[0]['shape']))
        np.copyto(in_buffers[1]['host'], right_np.reshape(in_buffers[1]['shape']))
        t_conv = time.perf_counter()

        if self._trt_stream is None:
            # Legacy TRT<=9 path: synchronous binding API.
            for io in in_buffers:
                self._cuda.memcpy_htod(io['dev'], io['host'])
            t_htod = time.perf_counter()
            self.trt_context.execute_v2(self.trt_bindings)
            t_exec = time.perf_counter()
            if self._use_gpu_depth:
                self._launch_depth_kernel(out_buffers, fx_baseline, None)
                self._cuda.memcpy_dtoh(self._depth_host, self._depth_dev)
            else:
                for io in out_buffers:
                    self._cuda.memcpy_dtoh(io['host'], io['dev'])
            t_dtoh = time.perf_counter()
        else:
            stream = self._trt_stream
            if not self._use_cuda_graph or self._trt_warmup_left > 0:
                self._enqueue_trt_async(in_buffers, out_buffers, fx_baseline)
                if self._trt_warmup_left > 0:
                    self._trt_warmup_left -= 1
            elif self._trt_graph_exec is None:
                try:
                    stream.begin_capture()
                    self._enqueue_trt_async(in_buffers, out_buffers, fx_baseline)
                    graph = stream.end_capture()
                    self._trt_graph_exec = graph.instantiate()
                    self._trt_graph_exec.launch(stream)
                    if self._use_gpu_depth:
                        self._depth_fx_baseline_captured = float(fx_baseline)
                    self.get_logger().info(
                        f'CUDA graph captured for TRT inference '
                        f'(gpu_depth={self._use_gpu_depth}).')
                except Exception as e:  # pycuda graph API varies; degrade gracefully
                    self._use_cuda_graph = False
                    self._trt_graph_exec = None
                    try:
                        stream.end_capture()  # in case capture was left open
                    except Exception:
                        pass
                    self.get_logger().warn(
                        f'CUDA graph capture failed ({e}); using plain async enqueue.')
                    self._enqueue_trt_async(in_buffers, out_buffers, fx_baseline)
            else:
                if (self._use_gpu_depth
                        and self._depth_fx_baseline_captured is not None
                        and abs(fx_baseline - self._depth_fx_baseline_captured) > 1e-3
                        and not getattr(self, '_warned_fx_change', False)):
                    self.get_logger().warn(
                        f'fx_baseline changed after CUDA graph capture '
                        f'({self._depth_fx_baseline_captured:.4f} -> {fx_baseline:.4f}); '
                        f'GPU depth will use the captured value.')
                    self._warned_fx_change = True
                self._trt_graph_exec.launch(stream)
            t_htod = t_conv  # async: H2D/exec/D2H overlap on the stream
            stream.synchronize()
            t_exec = t_dtoh = time.perf_counter()

        if self._use_gpu_depth:
            # Depth already computed on GPU and D2H'd into the pinned buffer.
            # Returning (depth, None, None) signals the caller to skip CPU mask.
            depth = self._depth_host
            t_out = time.perf_counter()
            self._last_infer_timing = {
                'conv': (t_conv - t0) * 1e3,
                'htod': (t_htod - t_conv) * 1e3,
                'exec': (t_exec - t_htod) * 1e3,
                'dtoh': (t_dtoh - t_exec) * 1e3,
                'out': (t_out - t_dtoh) * 1e3,
            }
            return depth, None, None

        disp = np.squeeze(out_buffers[0]['host']).astype(np.float32)
        occ = np.squeeze(out_buffers[1]['host']).astype(np.float32) if len(out_buffers) > 1 else None
        conf = np.squeeze(out_buffers[2]['host']).astype(np.float32) if len(out_buffers) > 2 else None
        t_out = time.perf_counter()
        self._last_infer_timing = {
            'conv': (t_conv - t0) * 1e3,
            'htod': (t_htod - t_conv) * 1e3,
            'exec': (t_exec - t_htod) * 1e3,
            'dtoh': (t_dtoh - t_exec) * 1e3,
            'out': (t_out - t_dtoh) * 1e3,
        }
        return disp, occ, conf


def main(args=None):
    rclpy.init(args=args)
    try:
        node = CustomStereoDepthNode()
    except SystemExit:
        rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
