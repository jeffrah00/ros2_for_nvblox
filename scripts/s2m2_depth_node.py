#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
S2M2 stereo-depth ROS 2 node.

Subscribes to a synchronized stereo pair (left/right rectified images) plus the
left CameraInfo, runs the S2M2 stereo-matching network to produce disparity,
converts disparity to metric depth, and publishes a depth image + CameraInfo
on the topics nvblox subscribes to. Supports both PyTorch pretrained weights
and a pre-exported TensorRT engine. Output depth is always restored to the
original input dimensions.
"""

import os
import sys
import time

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


def _to_3channel(img: np.ndarray) -> np.ndarray:
    """S2M2 expects a 3-channel image. Replicate mono into RGB."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


class S2M2DepthNode(Node):
    def __init__(self):
        super().__init__('s2m2_depth_node')
        self._declare_params()
        self._setup_backend()
        self._setup_io()
        self.bridge = CvBridge()

    # ------------------------------------------------------------------ params
    def _declare_params(self):
        self.declare_parameter('model_type', 'S')
        self.declare_parameter('num_refine', 1)
        self.declare_parameter('weights_path', '')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('left_topic', '/stereo/left/image_rect')
        self.declare_parameter('right_topic', '/stereo/right/image_rect')
        self.declare_parameter('camera_info_topic', '/stereo/left/camera_info')
        self.declare_parameter('output_depth_topic', '/s2m2/depth/image_rect_raw')
        self.declare_parameter('output_camera_info_topic', '/s2m2/depth/camera_info')
        self.declare_parameter('width', 0)
        self.declare_parameter('height', 0)
        self.declare_parameter('baseline_m', 0.05)
        self.declare_parameter('mask_occluded', True)
        self.declare_parameter('mask_low_confidence', True)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('fps_log_period', 5.0)

        self.cfg_width = int(self.get_parameter('width').value)
        self.cfg_height = int(self.get_parameter('height').value)
        if self.cfg_width and self.cfg_width % 32 != 0:
            self.get_logger().fatal(f'width={self.cfg_width} must be divisible by 32')
            raise SystemExit(1)
        if self.cfg_height and self.cfg_height % 32 != 0:
            self.get_logger().fatal(f'height={self.cfg_height} must be divisible by 32')
            raise SystemExit(1)
        self.baseline_m = float(self.get_parameter('baseline_m').value)
        self.mask_occluded = bool(self.get_parameter('mask_occluded').value)
        self.mask_low_conf = bool(self.get_parameter('mask_low_confidence').value)
        self._fps_log_period_s = float(self.get_parameter('fps_log_period').value)
        self._fps_window_start = time.perf_counter()
        self._fps_frame_count = 0
        self._fps_stage_acc = {}  # stage name -> accumulated ms within the window
        self._logged_first_frame = False
        self._prev_callback_end = None  # perf_counter at end of previous on_stereo

    # ----------------------------------------------------------------- backend
    def _setup_backend(self):
        engine = self.get_parameter('engine_path').value
        weights = self.get_parameter('weights_path').value
        device_str = self.get_parameter('device').value
        self.device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')

        if engine and os.path.isfile(engine):
            self._init_trt(engine)
            self.backend = 'trt'
            self.get_logger().info(f'S2M2 backend: TensorRT engine ({engine})')
        elif weights:
            from s2m2.core.utils.model_utils import load_model
            self.model = load_model(
                weights,
                self.get_parameter('model_type').value,
                True,  # no_negative_disparity
                int(self.get_parameter('num_refine').value),
                self.device)
            self.backend = 'torch'
            self.get_logger().info(
                f'S2M2 backend: PyTorch ({self.get_parameter("model_type").value}, '
                f'num_refine={self.get_parameter("num_refine").value})')
        else:
            self.get_logger().fatal('Need either engine_path or weights_path to be set.')
            raise SystemExit(1)

    def _init_trt(self, engine_path: str):
        if not (self.cfg_width and self.cfg_height):
            self.get_logger().fatal(
                'TensorRT path requires width and height to be set explicitly '
                '(both divisible by 32, matching the engine).')
            raise SystemExit(1)
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

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
            # TRT 10+ path: tensor-name API + execute_async_v3.
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                is_input = (self.trt_engine.get_tensor_mode(name)
                            == trt.TensorIOMode.INPUT)
                shape = tuple(self.trt_engine.get_tensor_shape(name))
                if -1 in shape and is_input:
                    self.trt_context.set_input_shape(name, (1, 3, H, W))
                    shape = tuple(self.trt_context.get_tensor_shape(name))
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                host_buf = np.empty(shape, dtype=dtype)
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                self.trt_context.set_tensor_address(name, int(dev_buf))
                self.trt_io.append({
                    'name': name, 'is_input': is_input, 'shape': shape,
                    'dtype': dtype, 'host': host_buf, 'dev': dev_buf,
                })
            self._trt_stream = cuda.Stream()
            self.trt_bindings = None
        else:
            # TRT <=9 path: binding-index API + execute_v2.
            self.trt_bindings = [None] * self.trt_engine.num_bindings
            for i in range(self.trt_engine.num_bindings):
                shape = tuple(self.trt_engine.get_binding_shape(i))
                is_input = self.trt_engine.binding_is_input(i)
                if -1 in shape and is_input:
                    self.trt_context.set_binding_shape(i, (1, 3, H, W))
                    shape = (1, 3, H, W)
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
                host_buf = np.empty(shape, dtype=dtype)
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                self.trt_bindings[i] = int(dev_buf)
                self.trt_io.append({
                    'name': self.trt_engine.get_binding_name(i),
                    'is_input': is_input, 'shape': shape, 'dtype': dtype,
                    'host': host_buf, 'dev': dev_buf,
                })
            self._trt_stream = None

        # Sanity-check: any input must accept (1,3,H,W).
        for io in self.trt_io:
            if io['is_input'] and io['shape'][-2:] != (H, W):
                self.get_logger().fatal(
                    f"Engine input shape {io['shape']} does not match "
                    f'configured (height={H}, width={W}). Re-export the engine.')
                raise SystemExit(1)

    # --------------------------------------------------------------------- io
    def _setup_io(self):
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        out_depth = self.get_parameter('output_depth_topic').value
        out_info = self.get_parameter('output_camera_info_topic').value

        left_sub = message_filters.Subscriber(self, Image, left_topic)
        right_sub = message_filters.Subscriber(self, Image, right_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, info_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_stereo)

        self.pub_depth = self.create_publisher(Image, out_depth, 5)
        self.pub_info = self.create_publisher(CameraInfo, out_info, 5)
        self.get_logger().info(
            f'subscribed: {left_topic}, {right_topic}, {info_topic}\n'
            f'publishing: {out_depth}, {out_info}')

    # --------------------------------------------------------------- callback
    def on_stereo(self, left_msg: Image, right_msg: Image, info_msg: CameraInfo):
        t_start = time.perf_counter()
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
        left = _to_3channel(left)
        right = _to_3channel(right)
        if left.dtype != np.uint8:
            left = left.astype(np.uint8)
            right = right.astype(np.uint8)
        t_decode = time.perf_counter()

        h_orig, w_orig = left.shape[:2]
        if self.cfg_width and self.cfg_height:
            # Resize mode.
            W, H = self.cfg_width, self.cfg_height
            sx, sy = W / w_orig, H / h_orig
            ox = oy = 0
            left_in = cv2.resize(left, (W, H), interpolation=cv2.INTER_LINEAR)
            right_in = cv2.resize(right, (W, H), interpolation=cv2.INTER_LINEAR)
            mode = 'resize'
        else:
            # Center-crop mode (default).
            H = (h_orig // 32) * 32
            W = (w_orig // 32) * 32
            if H == 0 or W == 0:
                self.get_logger().warn(
                    f'Input {h_orig}x{w_orig} is smaller than 32 in some dim; skipping frame.')
                return
            sx, sy = 1.0, 1.0
            ox = (w_orig - W) // 2
            oy = (h_orig - H) // 2
            left_in = left[oy:oy + H, ox:ox + W]
            right_in = right[oy:oy + H, ox:ox + W]
            mode = 'crop'

        t_pre = time.perf_counter()

        # Inference.
        if self.backend == 'torch':
            from s2m2.core.utils.model_utils import run_stereo_matching
            left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).to(self.device)
            right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                disp_t, occ_t, conf_t, _, _ = run_stereo_matching(
                    self.model, left_t, right_t, self.device, N_repeat=1)
            disp = disp_t.squeeze().detach().cpu().numpy().astype(np.float32)
            occ = occ_t.squeeze().detach().cpu().numpy()
            conf = conf_t.squeeze().detach().cpu().numpy()
        else:
            # TRT path takes numpy directly: no torch CUDA tensor (and no second
            # CUDA context) is created in this process.
            disp, occ, conf = self._trt_infer(left_in, right_in)
        t_infer = time.perf_counter()

        # Disparity -> depth at the inference resolution.
        fx_used = float(info_msg.k[0]) * sx
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 1e-3
        depth[valid] = (fx_used * self.baseline_m) / disp[valid]

        # Mask occluded / low-confidence pixels.
        # S2M2 model_utils semantics: occ == 0 means occluded; conf is binary,
        # 1 when disparity error < 4 px, else 0.
        mask = np.zeros_like(disp, dtype=bool)
        if self.mask_occluded:
            mask |= (occ < 0.5)
        if self.mask_low_conf:
            mask |= (conf < 0.5)
        depth[mask] = 0.0

        # Restore to original dimensions so depth aligns with the unmodified CameraInfo.
        if mode == 'resize':
            depth_full = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            depth_full = np.zeros((h_orig, w_orig), dtype=np.float32)
            depth_full[oy:oy + H, ox:ox + W] = depth
        t_post = time.perf_counter()

        depth_msg = self.bridge.cv2_to_imgmsg(depth_full, encoding='32FC1')
        depth_msg.header = left_msg.header
        self.pub_depth.publish(depth_msg)

        info_out = CameraInfo()
        info_out.header = left_msg.header
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
        self.pub_info.publish(info_out)
        t_pub = time.perf_counter()

        if not self._logged_first_frame:
            self._logged_first_frame = True
            self.get_logger().info(
                f'first frame: input {w_orig}x{h_orig} -> inference {W}x{H} ({mode}); '
                f'backend={self.backend}, device={self.device}')

        idle_ms = ((t_start - self._prev_callback_end) * 1e3
                   if self._prev_callback_end is not None else 0.0)
        self._prev_callback_end = t_pub
        self._update_fps({
            'idle': idle_ms,
            'decode': (t_decode - t_start) * 1e3,
            'pre': (t_pre - t_decode) * 1e3,
            'infer': (t_infer - t_pre) * 1e3,
            'post': (t_post - t_infer) * 1e3,
            'pub': (t_pub - t_post) * 1e3,
        })

    # ----------------------------------------------------------------- fps log
    def _update_fps(self, stage_ms: dict):
        if self._fps_log_period_s <= 0.0:
            return
        self._fps_frame_count += 1
        for name, ms in stage_ms.items():
            self._fps_stage_acc[name] = self._fps_stage_acc.get(name, 0.0) + ms
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed >= self._fps_log_period_s:
            count = self._fps_frame_count
            breakdown = ' | '.join(
                f'{name} {self._fps_stage_acc[name] / count:.1f}' for name in stage_ms)
            self.get_logger().info(
                f'FPS: {count / elapsed:.1f} ({count} frames in {elapsed:.1f}s) | '
                f'{breakdown} ms (avg)')
            self._fps_window_start = time.perf_counter()
            self._fps_frame_count = 0
            self._fps_stage_acc = {}

    # ----------------------------------------------------------------- trt fn
    def _trt_infer(self, left_in: np.ndarray, right_in: np.ndarray):
        # left_in/right_in are HxWx3 uint8; convert to NCHW in the engine's dtype.
        # (Engines exported expecting [0,1] float inputs should be re-exported or fed
        # a pre-scaled input; this passes uint8 cast to the engine dtype.)
        in_buffers = [io for io in self.trt_io if io['is_input']]
        out_buffers = [io for io in self.trt_io if not io['is_input']]

        # Heuristic mapping: the first input is left, second is right.
        left_np = left_in.transpose(2, 0, 1)[None].astype(in_buffers[0]['dtype'])
        right_np = right_in.transpose(2, 0, 1)[None].astype(in_buffers[1]['dtype'])
        np.copyto(in_buffers[0]['host'], left_np.reshape(in_buffers[0]['shape']))
        np.copyto(in_buffers[1]['host'], right_np.reshape(in_buffers[1]['shape']))

        for io in in_buffers:
            self._cuda.memcpy_htod(io['dev'], io['host'])
        if self._trt_use_tensor_api:
            self.trt_context.execute_async_v3(self._trt_stream.handle)
            self._trt_stream.synchronize()
        else:
            self.trt_context.execute_v2(self.trt_bindings)
        for io in out_buffers:
            self._cuda.memcpy_dtoh(io['host'], io['dev'])

        # Outputs assumed in the order: disparity, occlusion, confidence (matches
        # S2M2's export_tensorrt.py). Squeeze to (H, W).
        disp = np.squeeze(out_buffers[0]['host']).astype(np.float32)
        occ = np.squeeze(out_buffers[1]['host']).astype(np.float32)
        conf = np.squeeze(out_buffers[2]['host']).astype(np.float32)
        return disp, occ, conf


def main(args=None):
    rclpy.init(args=args)
    try:
        node = S2M2DepthNode()
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
