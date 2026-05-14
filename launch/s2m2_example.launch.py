# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
    args.add_arg('log_level', 'info', choices=[
                 'debug', 'info', 'warn'], cli=True)
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
        description='The  model type of PeopleSemSegNet (only used when mode:=people_segmentation).',
        cli=True)
    args.add_arg(
        'attach_to_container',
        'False',
        description='Add components to an existing component container.',
        cli=True)
    args.add_arg(
        'container_name',
        NVBLOX_CONTAINER_NAME,
        description='Name of the component container.')
    args.add_arg(
        'run_realsense',
        'True',
        description='Launch Realsense drivers')
    args.add_arg(
        'use_foxglove_whitelist',
        True,
        description='Disable visualization of bandwidth-heavy topics',
        cli=True)

    # ---- S2M2 custom-depth args ---------------------------------------------
    args.add_arg('depth_source', 's2m2', choices=['s2m2', 'realsense'],
                 description='Which depth source feeds nvblox.', cli=True)
    args.add_arg('s2m2_model_type', 'S', choices=['S', 'M', 'L', 'XL'],
                 description='S2M2 model variant.', cli=True)
    args.add_arg('s2m2_num_refine', 1,
                 description='S2M2 local iterative refinement count.', cli=True)
    args.add_arg('s2m2_weights_path', '',
                 description='Directory with S2M2 PyTorch weights (used if engine_path is empty).',
                 cli=True)
    args.add_arg('s2m2_engine_path', '',
                 description='TensorRT .engine path. If set, takes priority over weights_path.',
                 cli=True)
    # Defaults match RealSense D435/D435i IR streams under nvblox_examples_bringup's
    # /camera0 namespace. Override these for any other stereo source.
    args.add_arg('s2m2_left_topic', '/camera0/infra1/image_rect_raw',
                 description='Left rectified stereo image. D435i IR1 by default.', cli=True)
    args.add_arg('s2m2_right_topic', '/camera0/infra2/image_rect_raw',
                 description='Right rectified stereo image. D435i IR2 by default.', cli=True)
    args.add_arg('s2m2_camera_info_topic', '/camera0/infra1/camera_info',
                 description='CameraInfo for the left image. D435i IR1 by default.', cli=True)
    args.add_arg('s2m2_output_depth_topic', '/s2m2/depth/image_rect_raw',
                 description='Where the S2M2 node publishes depth. nvblox is remapped here.',
                 cli=True)
    args.add_arg('s2m2_output_camera_info_topic', '/s2m2/depth/camera_info',
                 description='Where the S2M2 node publishes the matching CameraInfo.',
                 cli=True)
    args.add_arg('s2m2_width', 0,
                 description='Inference width. 0 = crop to nearest /32. Must be divisible by 32 if set.',
                 cli=True)
    args.add_arg('s2m2_height', 0,
                 description='Inference height. 0 = crop to nearest /32. Must be divisible by 32 if set.',
                 cli=True)
    args.add_arg('s2m2_baseline_m', 0.05,
                 description='Stereo baseline in meters (left CameraInfo does not carry it).',
                 cli=True)
    args.add_arg('s2m2_mask_occluded', 'True',
                 description='Zero out depth where the S2M2 occlusion map is 0 (occluded).',
                 cli=True)
    args.add_arg('s2m2_mask_low_confidence', 'True',
                 description='Zero out depth where the S2M2 confidence map is 0 '
                             '(disparity error >= 4 px).',
                 cli=True)
    args.add_arg('s2m2_device', 'cuda', choices=['cuda', 'cpu'], cli=True)

    actions = args.get_launch_actions()

    # Globally set use_sim_time if we're running from bag or sim
    actions.append(
        SetParameter('use_sim_time', True, condition=IfCondition(lu.is_valid(args.rosbag))))

    # Single or Multi-realsense
    is_multi_cam = UnlessCondition(lu.is_equal(args.num_cameras, '1'))
    camera_mode = lu.if_else_substitution(
        lu.is_equal(args.num_cameras, '1'),
        str(NvbloxCamera.realsense),
        str(NvbloxCamera.multi_realsense)
    )
    # Only up to 4 Realsenses is supported.
    actions.append(
        lu.assert_condition(
            'Up to 4 cameras have been tested! num_cameras must be less than 5.',
            IfCondition(PythonExpression(['int("', args.num_cameras, '") > 4']))),
    )

    # The RealSense driver still runs in the s2m2 path because we use its IR1/IR2
    # streams as the stereo input to S2M2.
    run_rs_driver = UnlessCondition(
        OrSubstitution(lu.is_valid(args.rosbag), lu.is_false(args.run_realsense)))
    # Realsense
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

    # Visual SLAM
    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/perception/vslam.launch.py',
            launch_arguments={
                'container_name': args.container_name,
                'camera': camera_mode,
            },
            # Delay for 1 second to make sure that the static topics from the rosbag are published.
            delay=1.0,
        ))
    # People detection for multi-RS
    camera_namespaces = ['camera0', 'camera1', 'camera2', 'camera3']
    camera_input_topics = []
    input_camera_info_topics= []
    output_resized_image_topics = []
    output_resized_camera_info_topics = []
    for ns in camera_namespaces:
        camera_input_topics.append(f'/{ns}/color/image_raw')
        input_camera_info_topics.append(f'/{ns}/color/camera_info')
        output_resized_image_topics.append(f'/{ns}/segmentation/image_resized')
        output_resized_camera_info_topics.append(f'/{ns}/segmentation/camera_info_resized')

    # People segmentation
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
                # fixing rosbag replay dropping fps
                'one_container_per_camera': True
            },
            condition=IfCondition(lu.has_substring(args.mode, NvbloxMode.people_segmentation))))

    # People detection
    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/perception/detection.launch.py',
            launch_arguments={
                'namespace_list': camera_namespaces,
                'input_topic_list': camera_input_topics,
                'num_cameras': args.num_cameras,
                'container_name': args.container_name,
                # fixing rosbag replay dropping fps
                'one_container_per_camera': True
            },
            condition=IfCondition(lu.has_substring(args.mode, NvbloxMode.people_detection))))

    # Nvblox. When depth_source=s2m2, remap the depth topics to the S2M2 output
    # topics so nvblox subscribes to S2M2's depth instead of the RealSense's.
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
            SetRemap(src='camera_0/depth/image', dst=args.s2m2_output_depth_topic),
            SetRemap(src='camera_0/depth/camera_info', dst=args.s2m2_output_camera_info_topic),
            _nvblox_include(),
        ],
        condition=LaunchConfigurationEquals('depth_source', 's2m2')))

    actions.append(GroupAction(
        actions=[_nvblox_include()],
        condition=LaunchConfigurationEquals('depth_source', 'realsense')))

    # TF transforms for multi-realsense
    actions.append(
        lu.add_robot_description(robot_calibration_path=args.multicam_urdf_path,
                                 condition=is_multi_cam)
    )

    # Play ros2bag
    actions.append(
        lu.play_rosbag(
            bag_path=args.rosbag,
            additional_bag_play_args=args.rosbag_args,
            condition=IfCondition(lu.is_valid(args.rosbag))))

    # Visualization
    actions.append(
        lu.include(
            'nvblox_examples_bringup',
            'launch/visualization/visualization.launch.py',
            launch_arguments={
                'mode': args.mode,
                'camera': camera_mode,
                'use_foxglove_whitelist': args.use_foxglove_whitelist,
            }))

    # S2M2 stereo-depth node. Run as a plain Python script so this repo stays
    # package-free; ROS params are forwarded via --ros-args. Empty string params
    # are skipped because `-p name:=` (no value) is rejected by rcl.
    s2m2_script = os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 's2m2_depth_node.py')

    def _build_s2m2_proc(context, *_):
        param_names = [
            'model_type', 'num_refine', 'weights_path', 'engine_path',
            'left_topic', 'right_topic', 'camera_info_topic',
            'output_depth_topic', 'output_camera_info_topic',
            'width', 'height', 'baseline_m', 'mask_occluded',
            'mask_low_confidence', 'device',
        ]
        cmd = ['python3', s2m2_script, '--ros-args']
        for name in param_names:
            value = LaunchConfiguration(f's2m2_{name}').perform(context)
            if value == '':
                continue
            cmd.extend(['-p', f'{name}:={value}'])
        return [ExecuteProcess(cmd=cmd, output='screen')]

    actions.append(GroupAction(
        actions=[OpaqueFunction(function=_build_s2m2_proc)],
        condition=LaunchConfigurationEquals('depth_source', 's2m2')))

    # Container
    # NOTE: By default (attach_to_container:=False) we launch a container which all nodes are
    # added to, however, we expose the option to not launch a container, and instead attach to
    # an already running container. The reason for this is that when running live on multiple
    # realsenses we have experienced unreliability in the bringup of multiple realsense drivers.
    # To (partially) mitigate this issue the suggested workflow for multi-realsenses is to:
    # 1. Launch RS (cameras & splitter) and start a component_container
    # 2. Launch nvblox + cuvslam and attached to the above running component container

    actions.append(
        lu.component_container(
            NVBLOX_CONTAINER_NAME, condition=UnlessCondition(args.attach_to_container),
            log_level=args.log_level))

    return LaunchDescription(actions)
