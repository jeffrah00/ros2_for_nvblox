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

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
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

        H, W = self.cfg_height, self.cfg_width
        self.trt_io = []
        self.trt_bindings = [None] * self.trt_engine.num_bindings
        for i in range(self.trt_engine.num_bindings):
            shape = tuple(self.trt_engine.get_binding_shape(i))
            if -1 in shape and self.trt_engine.binding_is_input(i):
                self.trt_context.set_binding_shape(i, (1, 3, H, W))
                shape = (1, 3, H, W)
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
            host_buf = np.empty(shape, dtype=dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            self.trt_bindings[i] = int(dev_buf)
            self.trt_io.append({
                'name': self.trt_engine.get_binding_name(i),
                'is_input': self.trt_engine.binding_is_input(i),
                'shape': shape,
                'dtype': dtype,
                'host': host_buf,
                'dev': dev_buf,
            })
        self._cuda = cuda

        n_in = sum(1 for io in self.trt_io if io['is_input'])
        n_out = sum(1 for io in self.trt_io if not io['is_input'])
        if n_in != 2:
            self.get_logger().fatal(f'Engine has {n_in} inputs; expected 2 (left, right).')
            raise SystemExit(1)
        if n_out < 1 or n_out > 3:
            self.get_logger().fatal(
                f'Engine has {n_out} outputs; expected 1 (disp), 2 (disp, occ), or 3 (disp, occ, conf).')
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
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
        left = _to_3channel(left)
        right = _to_3channel(right)
        if left.dtype != np.uint8:
            left = left.astype(np.uint8)
            right = right.astype(np.uint8)

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

        # Inference.
        if self.backend == 'onnx':
            disp, occ, conf = self._onnx_infer(left_np, right_np)
        else:
            disp, occ, conf = self._trt_infer(left_np, right_np)

        # Disparity -> depth at the inference resolution.
        fx_used = float(info_msg.k[0]) * sx
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 1e-3
        depth[valid] = (fx_used * self.baseline_m) / disp[valid]

        # Optional masking.
        mask = None
        if occ is not None:
            mask = (occ > 0.5)
        if conf is not None and self.conf_thr > 0.0:
            cmask = (conf < self.conf_thr)
            mask = cmask if mask is None else (mask | cmask)
        if mask is not None:
            depth[mask] = 0.0

        # Restore to original dimensions.
        if mode == 'resize':
            depth_full = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            depth_full = np.zeros((h_orig, w_orig), dtype=np.float32)
            depth_full[oy:oy + H, ox:ox + W] = depth

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

    # ---------------------------------------------------------------- onnx fn
    def _onnx_infer(self, left_np: np.ndarray, right_np: np.ndarray):
        outs = self.ort_sess.run(
            self.ort_output_names,
            {self.ort_input_names[0]: left_np,
             self.ort_input_names[1]: right_np})
        disp = np.squeeze(outs[0]).astype(np.float32)
        occ = np.squeeze(outs[1]).astype(np.float32) if len(outs) > 1 else None
        conf = np.squeeze(outs[2]).astype(np.float32) if len(outs) > 2 else None
        return disp, occ, conf

    # ----------------------------------------------------------------- trt fn
    def _trt_infer(self, left_np: np.ndarray, right_np: np.ndarray):
        in_buffers = [io for io in self.trt_io if io['is_input']]
        out_buffers = [io for io in self.trt_io if not io['is_input']]

        left_np = left_np.astype(in_buffers[0]['dtype'])
        right_np = right_np.astype(in_buffers[1]['dtype'])
        np.copyto(in_buffers[0]['host'], left_np.reshape(in_buffers[0]['shape']))
        np.copyto(in_buffers[1]['host'], right_np.reshape(in_buffers[1]['shape']))

        for io in in_buffers:
            self._cuda.memcpy_htod(io['dev'], io['host'])
        self.trt_context.execute_v2(self.trt_bindings)
        for io in out_buffers:
            self._cuda.memcpy_dtoh(io['host'], io['dev'])

        disp = np.squeeze(out_buffers[0]['host']).astype(np.float32)
        occ = np.squeeze(out_buffers[1]['host']).astype(np.float32) if len(out_buffers) > 1 else None
        conf = np.squeeze(out_buffers[2]['host']).astype(np.float32) if len(out_buffers) > 2 else None
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
