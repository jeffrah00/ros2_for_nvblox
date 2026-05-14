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
        self.declare_parameter('stall_warn_ms', 250.0)
        self.declare_parameter('use_cuda_graph', True)
        self.declare_parameter('use_sensor_qos', False)
        self.declare_parameter('process_every_n', 1)
        # Two extra subscriptions that re-deserialize the left image and the
        # depth output every frame purely to count Hz. Off by default; the same
        # numbers are available from `ros2 topic hz`.
        self.declare_parameter('enable_rate_probes', False)

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
            # TRT <=9 path: binding-index API + execute_v2.
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

        # Sanity-check: any input must accept (1,3,H,W).
        for io in self.trt_io:
            if io['is_input'] and io['shape'][-2:] != (H, W):
                self.get_logger().fatal(
                    f"Engine input shape {io['shape']} does not match "
                    f'configured (height={H}, width={W}). Re-export the engine.')
                raise SystemExit(1)

        # CUDA-graph capture of H2D->execute->D2H (TRT 10+ tensor API only).
        # Valid because the page-locked host buffers and device buffers above
        # have fixed addresses for the node's lifetime; only their contents change.
        self._use_cuda_graph = (bool(self.get_parameter('use_cuda_graph').value)
                                and self._trt_use_tensor_api)
        self._trt_graph_exec = None
        self._trt_warmup_left = 2  # TRT needs a couple of real runs before capture

    # --------------------------------------------------------------------- io
    def _setup_io(self):
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        out_depth = self.get_parameter('output_depth_topic').value
        out_info = self.get_parameter('output_camera_info_topic').value

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

        left_sub = message_filters.Subscriber(self, Image, left_topic, **sub_kw)
        right_sub = message_filters.Subscriber(self, Image, right_topic, **sub_kw)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic, **sub_kw)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, info_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_stereo)

        # Lightweight extra subscriptions purely for rate measurement, independent
        # of `ros2 topic hz` (a separate subscriber): one on the left input image
        # (true in-process receive rate) and one on our own depth output (rate at
        # which published depth actually comes back through the transport).
        # Gated because each fires a Python callback that re-deserializes a full
        # Image every frame -- 1-3 ms/frame of pure overhead at typical rates.
        self._rate_probes = bool(self.get_parameter('enable_rate_probes').value)
        self.pub_depth = self.create_publisher(Image, out_depth, pub_qos)
        self.pub_info = self.create_publisher(CameraInfo, out_info, pub_qos)
        if self._rate_probes:
            self._incoming_sub = self.create_subscription(
                Image, left_topic, self._incoming_cb, plain_qos)
            self._outgoing_sub = self.create_subscription(
                Image, out_depth, self._outgoing_cb, plain_qos)
        self.get_logger().info(
            f'subscribed: {left_topic}, {right_topic}, {info_topic}\n'
            f'publishing: {out_depth}, {out_info}')

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

        # Inference. Each branch fills self._last_infer_timing with sub-stage ms.
        if self.backend == 'torch':
            from s2m2.core.utils.model_utils import run_stereo_matching
            ta = time.perf_counter()
            left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).to(self.device)
            right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).to(self.device)
            tb = time.perf_counter()
            with torch.inference_mode():
                disp_t, occ_t, conf_t, _, _ = run_stereo_matching(
                    self.model, left_t, right_t, self.device, N_repeat=1)
            tc = time.perf_counter()
            disp = disp_t.squeeze().detach().cpu().numpy().astype(np.float32)
            occ = occ_t.squeeze().detach().cpu().numpy()
            conf = conf_t.squeeze().detach().cpu().numpy()
            td = time.perf_counter()
            self._last_infer_timing = {
                'up': (tb - ta) * 1e3, 'exec': (tc - tb) * 1e3, 'down': (td - tc) * 1e3,
            }
        else:
            # TRT path takes numpy directly: no torch CUDA tensor (and no second
            # CUDA context) is created in this process.
            disp, occ, conf = self._trt_infer(left_in, right_in)
        t_infer = time.perf_counter()

        # Disparity -> depth at the inference resolution. One pass: build the
        # valid mask (incl. occ/conf) once, then divide into a pre-zeroed buffer.
        # Same final result as the previous gather/scatter divide + |= masking.
        # S2M2 model_utils semantics: occ == 0 means occluded; conf is binary,
        # 1 when disparity error < 4 px, else 0.
        fx_used = float(info_msg.k[0]) * sx
        valid = disp > 1e-3
        if self.mask_occluded:
            valid &= (occ >= 0.5)
        if self.mask_low_conf:
            valid &= (conf >= 0.5)
        if self._depth_buf is None or self._depth_buf.shape != disp.shape:
            self._depth_buf = np.zeros(disp.shape, dtype=np.float32)
        depth = self._depth_buf
        depth.fill(0.0)
        np.divide(fx_used * self.baseline_m, disp, out=depth, where=valid)

        # Restore to original dimensions so depth aligns with the unmodified CameraInfo.
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
        depth_msg.header = left_msg.header
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
        self._info_out.header = left_msg.header
        self.pub_info.publish(self._info_out)
        t_pub = time.perf_counter()

        if not self._logged_first_frame:
            self._logged_first_frame = True
            self.get_logger().info(
                f'first frame: input {w_orig}x{h_orig} -> inference {W}x{H} ({mode}); '
                f'backend={self.backend}, device={self.device}')

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
            if self._rate_probes:
                rate_str = (f'in {self._incoming_count / elapsed:.1f} Hz, '
                            f'out {self._outgoing_count / elapsed:.1f} Hz, ')
            else:
                rate_str = ''
            self.get_logger().info(
                f'FPS: {count / elapsed:.1f} ({rate_str}{count} frames in {elapsed:.1f}s) | '
                f'{breakdown} ms (avg/max)')
            self._fps_window_start = time.perf_counter()
            self._fps_frame_count = 0
            self._fps_stage_acc = {}
            self._fps_stage_max = {}
            self._incoming_count = 0
            self._outgoing_count = 0

    # ----------------------------------------------------------------- trt fn
    def _enqueue_trt_async(self, in_buffers, out_buffers):
        """H2D copies -> execute -> D2H copies, all enqueued on self._trt_stream."""
        stream = self._trt_stream
        for io in in_buffers:
            self._cuda.memcpy_htod_async(io['dev'], io['host'], stream)
        self.trt_context.execute_async_v3(stream.handle)
        for io in out_buffers:
            self._cuda.memcpy_dtoh_async(io['host'], io['dev'], stream)

    def _trt_infer(self, left_in: np.ndarray, right_in: np.ndarray):
        # left_in/right_in are HxWx3 uint8; convert to NCHW in the engine's dtype.
        # (Engines exported expecting [0,1] float inputs should be re-exported or fed
        # a pre-scaled input; this passes uint8 cast to the engine dtype.)
        in_buffers = [io for io in self.trt_io if io['is_input']]
        out_buffers = [io for io in self.trt_io if not io['is_input']]
        t0 = time.perf_counter()

        # Heuristic mapping: the first input is left, second is right. transpose()
        # + reshape() are views; np.copyto does the strided copy AND the dtype cast
        # straight into the (page-locked) host buffer in one pass -- bit-identical
        # to the previous astype/reshape/copyto chain, just fewer temporaries.
        np.copyto(in_buffers[0]['host'],
                  left_in.transpose(2, 0, 1).reshape(in_buffers[0]['shape']))
        np.copyto(in_buffers[1]['host'],
                  right_in.transpose(2, 0, 1).reshape(in_buffers[1]['shape']))
        t_conv = time.perf_counter()

        if self._trt_stream is None:
            # Legacy TRT<=9 path: synchronous binding API, unchanged behaviour.
            for io in in_buffers:
                self._cuda.memcpy_htod(io['dev'], io['host'])
            t_htod = time.perf_counter()
            self.trt_context.execute_v2(self.trt_bindings)
            t_exec = time.perf_counter()
            for io in out_buffers:
                self._cuda.memcpy_dtoh(io['host'], io['dev'])
            t_dtoh = time.perf_counter()
        else:
            stream = self._trt_stream
            if not self._use_cuda_graph or self._trt_warmup_left > 0:
                self._enqueue_trt_async(in_buffers, out_buffers)
                if self._trt_warmup_left > 0:
                    self._trt_warmup_left -= 1
            elif self._trt_graph_exec is None:
                try:
                    stream.begin_capture()
                    self._enqueue_trt_async(in_buffers, out_buffers)
                    graph = stream.end_capture()
                    self._trt_graph_exec = graph.instantiate()
                    self._trt_graph_exec.launch(stream)
                    self.get_logger().info('CUDA graph captured for TRT inference.')
                except Exception as e:  # pycuda graph API varies; degrade gracefully
                    self._use_cuda_graph = False
                    self._trt_graph_exec = None
                    try:
                        stream.end_capture()  # in case capture was left open
                    except Exception:
                        pass
                    self.get_logger().warn(
                        f'CUDA graph capture failed ({e}); using plain async enqueue.')
                    self._enqueue_trt_async(in_buffers, out_buffers)
            else:
                self._trt_graph_exec.launch(stream)
            t_htod = t_conv  # async: H2D/exec/D2H overlap on the stream
            stream.synchronize()
            t_exec = t_dtoh = time.perf_counter()

        # Outputs assumed in the order: disparity, occlusion, confidence (matches
        # S2M2's export_tensorrt.py). Squeeze to (H, W) -- view of the pinned
        # buffer. Only cast when the engine emits something other than float32;
        # otherwise the downstream np.divide(out=depth) reads the pinned buffer
        # directly and finishes before the next frame's H2D begins (the stream
        # has been synchronized above and this callback is single-threaded).
        disp = np.squeeze(out_buffers[0]['host'])
        occ = np.squeeze(out_buffers[1]['host'])
        conf = np.squeeze(out_buffers[2]['host'])
        if disp.dtype != np.float32: disp = disp.astype(np.float32)
        if occ.dtype  != np.float32: occ  = occ.astype(np.float32)
        if conf.dtype != np.float32: conf = conf.astype(np.float32)
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
