# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Launch nvblox with a custom stereo-depth model (TorchScript or TensorRT).

The companion `custom_depth_node.py` is the generic counterpart of
`s2m2_depth_node.py`: it accepts any model that takes a stereo pair and
returns disparity (optionally plus occlusion / confidence). Stereo defaults
target the RealSense D435i IR pair; override the topic args for any other
source.
"""

import os

from isaac_ros_launch_utils.all_types import *
import isaac_ros_launch_utils as lu

from launch.actions import GroupAction, ExecuteProcess, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap

from nvblox_ros_python_utils.nvblox_launch_utils import NvbloxMode, NvbloxCamera, NvbloxPeopleSegmentation
from nvblox_ros_python_utils.nvblox_constants import NVBLOX_CONTAINER_NAME


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg(
        'rosbag', 'None', description='Path to rosbag (running on sensor if not set).', cli=True)
    args.add_arg('rosbag_args', '',
                 description='Additional args for ros2 bag play.', cli=True)
    args.add_arg('log_level', 'info', choices=['debug', 'info', 'warn'], cli=True)
    args.add_arg('num_cameras', 1,
                 description='How many cameras to use.', cli=True)
    args.add_arg('camera_serial_numbers', '',
                 description='List of the serial no of the extra cameras. (comma separated)',
                 cli=True)
    args.add_arg(
        'multicam_urdf_path',
        lu.get_path('nvblox_examples_bringup',
                    'config/urdf/4_realsense_carter_example_calibration.urdf.xacro'),
        description='Path to a URDF file describing the camera rig extrinsics. Only used in multicam.',
        cli=True)
    args.add_arg(
        'mode',
        default=NvbloxMode.static,
        choices=NvbloxMode.names(),
        description='The nvblox mode.',
        cli=True)
    args.add_arg(
        'people_segmentation',
        default=NvbloxPeopleSegmentation.peoplesemsegnet_vanilla,
        choices=[
            str(NvbloxPeopleSegmentation.peoplesemsegnet_vanilla),
            str(NvbloxPeopleSegmentation.peoplesemsegnet_shuffleseg)
        ],
        description='The model type of PeopleSemSegNet (only used when mode:=people_segmentation).',
        cli=True)
    args.add_arg('attach_to_container', 'False',
                 description='Add components to an existing component container.', cli=True)
    args.add_arg('container_name', NVBLOX_CONTAINER_NAME,
                 description='Name of the component container.')
    args.add_arg('run_realsense', 'True', description='Launch Realsense drivers')
    args.add_arg('use_foxglove_whitelist', True,
                 description='Disable visualization of bandwidth-heavy topics', cli=True)
    args.add_arg('rviz_config', '',
                 description='Optional .rviz file. If set, forwarded to '
                             'nvblox_examples_bringup visualization.launch.py as rviz_config:=. '
                             'Leave empty to use the upstream default.',
                 cli=True)

    # ---- Custom-depth args -------------------------------------------------
    args.add_arg('depth_source', 'custom', choices=['custom', 'realsense'],
                 description='Which depth source feeds nvblox.', cli=True)
    args.add_arg('custom_onnx_path', '',
                 description='ONNX .onnx path (used if engine_path is empty).',
                 cli=True)
    args.add_arg('custom_engine_path', '',
                 description='TensorRT .engine path. If set, takes priority over onnx_path.',
                 cli=True)
    # Defaults match RealSense D435/D435i IR streams under the /camera0 namespace.
    args.add_arg('custom_left_topic', '/camera0/infra1/image_rect_raw',
                 description='Left rectified stereo image. D435i IR1 by default.', cli=True)
    args.add_arg('custom_right_topic', '/camera0/infra2/image_rect_raw',
                 description='Right rectified stereo image. D435i IR2 by default.', cli=True)
    args.add_arg('custom_camera_info_topic', '/camera0/infra1/camera_info',
                 description='CameraInfo for the left image. D435i IR1 by default.', cli=True)
    args.add_arg('custom_output_depth_topic', '/custom_depth/depth/image_rect_raw',
                 description='Where the custom node publishes depth. nvblox is remapped here.',
                 cli=True)
    args.add_arg('custom_output_camera_info_topic', '/custom_depth/depth/camera_info',
                 description='Where the custom node publishes the matching CameraInfo.',
                 cli=True)
    args.add_arg('custom_width', 0,
                 description='Inference width. 0 = crop to nearest /32. Must be divisible by 32 if set.',
                 cli=True)
    args.add_arg('custom_height', 0,
                 description='Inference height. 0 = crop to nearest /32. Must be divisible by 32 if set.',
                 cli=True)
    args.add_arg('custom_baseline_m', 0.05,
                 description='Stereo baseline in meters (left CameraInfo does not carry it).',
                 cli=True)
    args.add_arg('custom_confidence_threshold', 0.0,
                 description='Zero out depth where conf < threshold. 0 disables masking.',
                 cli=True)
    args.add_arg('custom_input_scale', 1.0 / 255.0,
                 description='Multiplier applied to uint8 input before inference. '
                             'Default 1/255 -> float [0,1]. Set to 1.0 for raw uint8 as float32.',
                 cli=True)
    args.add_arg('custom_device', 'cuda', choices=['cuda', 'cpu'], cli=True)

    actions = args.get_launch_actions()

    actions.append(
        SetParameter('use_sim_time', True, condition=IfCondition(lu.is_valid(args.rosbag))))

    is_multi_cam = UnlessCondition(lu.is_equal(args.num_cameras, '1'))
    camera_mode = lu.if_else_substitution(
        lu.is_equal(args.num_cameras, '1'),
        str(NvbloxCamera.realsense),
        str(NvbloxCamera.multi_realsense)
    )
    actions.append(
        lu.assert_condition(
            'Up to 4 cameras have been tested! num_cameras must be less than 5.',
            IfCondition(PythonExpression(['int("', args.num_cameras, '") > 4']))),
    )

    # The RealSense driver still runs in the custom path because we use its IR1/IR2
    # streams as the stereo input to the model.
    run_rs_driver = UnlessCondition(
        OrSubstitution(lu.is_valid(args.rosbag), lu.is_false(args.run_realsense)))
    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/sensors/realsense.launch.py',
            launch_arguments={
                'container_name': args.container_name,
                'camera_serial_numbers': args.camera_serial_numbers,
                'num_cameras': args.num_cameras,
            },
            condition=run_rs_driver))

    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/perception/vslam.launch.py',
            launch_arguments={
                'container_name': args.container_name,
                'camera': camera_mode,
            },
            delay=1.0,
        ))

    camera_namespaces = ['camera0', 'camera1', 'camera2', 'camera3']
    camera_input_topics = []
    input_camera_info_topics = []
    output_resized_image_topics = []
    output_resized_camera_info_topics = []
    for ns in camera_namespaces:
        camera_input_topics.append(f'/{ns}/color/image_raw')
        input_camera_info_topics.append(f'/{ns}/color/camera_info')
        output_resized_image_topics.append(f'/{ns}/segmentation/image_resized')
        output_resized_camera_info_topics.append(f'/{ns}/segmentation/camera_info_resized')

    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/perception/segmentation.launch.py',
            launch_arguments={
                'container_name': args.container_name,
                'people_segmentation': args.people_segmentation,
                'namespace_list': camera_namespaces,
                'input_topic_list': camera_input_topics,
                'input_camera_info_topic_list': input_camera_info_topics,
                'output_resized_image_topic_list': output_resized_image_topics,
                'output_resized_camera_info_topic_list': output_resized_camera_info_topics,
                'num_cameras': args.num_cameras,
                'one_container_per_camera': True
            },
            condition=IfCondition(lu.has_substring(args.mode, NvbloxMode.people_segmentation))))

    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/perception/detection.launch.py',
            launch_arguments={
                'namespace_list': camera_namespaces,
                'input_topic_list': camera_input_topics,
                'num_cameras': args.num_cameras,
                'container_name': args.container_name,
                'one_container_per_camera': True
            },
            condition=IfCondition(lu.has_substring(args.mode, NvbloxMode.people_detection))))

    # nvblox: when depth_source=custom, route its depth subscription to the
    # custom node's output topic via SetRemap.
    def _nvblox_include():
        return lu.include(
            'nvblox_examples_bringup',
            'launch/perception/nvblox.launch.py',
            launch_arguments={
                'container_name': args.container_name,
                'mode': args.mode,
                'camera': camera_mode,
                'num_cameras': args.num_cameras,
            })

    actions.append(GroupAction(
        actions=[
            SetRemap(src='camera_0/depth/image', dst=args.custom_output_depth_topic),
            SetRemap(src='camera_0/depth/camera_info', dst=args.custom_output_camera_info_topic),
            _nvblox_include(),
        ],
        condition=LaunchConfigurationEquals('depth_source', 'custom')))

    actions.append(GroupAction(
        actions=[_nvblox_include()],
        condition=LaunchConfigurationEquals('depth_source', 'realsense')))

    actions.append(
        lu.add_robot_description(robot_calibration_path=args.multicam_urdf_path,
                                 condition=is_multi_cam)
    )

    actions.append(
        lu.play_rosbag(
            bag_path=args.rosbag,
            additional_bag_play_args=args.rosbag_args,
            condition=IfCondition(lu.is_valid(args.rosbag))))

    # Visualization. rviz_config is forwarded only when non-empty so we don't
    # override the upstream default with an empty path.
    def _build_visualization(context, *_):
        viz_args = {
            'mode': args.mode,
            'camera': camera_mode,
            'use_foxglove_whitelist': args.use_foxglove_whitelist,
        }
        rviz_cfg = LaunchConfiguration('rviz_config').perform(context)
        if rviz_cfg:
            viz_args['rviz_config'] = rviz_cfg
        return [lu.include(
            'nvblox_examples_bringup',
            'launch/visualization/visualization.launch.py',
            launch_arguments=viz_args)]
    actions.append(OpaqueFunction(function=_build_visualization))

    # Custom stereo-depth node. Build the cmd at evaluation time and skip empty
    # params (rcl rejects `-p name:=` with no value).
    custom_script = os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'custom_depth_node.py')

    def _build_custom_proc(context, *_):
        param_names = [
            'onnx_path', 'engine_path',
            'left_topic', 'right_topic', 'camera_info_topic',
            'output_depth_topic', 'output_camera_info_topic',
            'width', 'height',
            'baseline_m', 'confidence_threshold', 'input_scale',
            'device',
        ]
        cmd = ['python3', custom_script, '--ros-args']
        for name in param_names:
            value = LaunchConfiguration(f'custom_{name}').perform(context)
            if value == '':
                continue
            cmd.extend(['-p', f'{name}:={value}'])
        return [ExecuteProcess(cmd=cmd, output='screen')]

    actions.append(GroupAction(
        actions=[OpaqueFunction(function=_build_custom_proc)],
        condition=LaunchConfigurationEquals('depth_source', 'custom')))

    actions.append(
        lu.component_container(
            NVBLOX_CONTAINER_NAME, condition=UnlessCondition(args.attach_to_container),
            log_level=args.log_level))

    return LaunchDescription(actions)
