import cv2

import numpy as np
import numpy.typing as npt

import FramesViewer.utils as fv_utils

from contact_graspnet_pytorch.wrapper import ContactGraspNetWrapper
from pollen_manipulation.reachy_1.reachability import is_pose_reachable, read_angle_limits
from pollen_manipulation.utils import normalize_pose

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

import time
from typing import Any, Dict, List, Tuple


class Reachy1ManipulationAPI:
    def __init__(self, reachy_ip: str, T_world_cam: npt.NDArray[np.float32], K_cam_left: npt.NDArray[np.float32]):
        self.reachy = ReachySDK(host=reachy_ip)

        self.T_world_cam = T_world_cam
        self.K_cam_left = K_cam_left

        self.right_start_pose = fv_utils.make_pose([0.20, -0.24, -0.28], [0, -90, 0])
        self.left_start_pose = fv_utils.make_pose([0.20, 0.24, -0.28], [0, -90, 0])
        self.right_joint_start_pose = self.reachy.r_arm.inverse_kinematics(self.right_start_pose)
        self.left_joint_start_pose = self.reachy.l_arm.inverse_kinematics(self.left_start_pose)

        self.angle_limits = read_angle_limits(self.reachy)

        self.grasp_net = ContactGraspNetWrapper()

    def grasp_object(self, object_info: Dict[str, Any], left: bool = False) -> bool:  # w/ vision
        position = object_info["position"]
        rgb = object_info["rgb"]
        mask = object_info["mask"]
        depth = object_info["depth"]

        if len(position) == 0:
            return False

        grasp_pose, _ = self.get_reachable_grasp_poses(rgb, depth, mask)
        if len(grasp_pose) == 0:
            return False

        print("GRASP POSE: ", grasp_pose[0])

        grasp_success = self.execute_grasp(grasp_pose[0])
        return grasp_success

    def get_reachable_grasp_poses(
        self,
        rgb: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        left: bool = False,
        visualize: bool = False,
    ) -> Tuple[List[npt.NDArray[np.float32]], List[np.float32]]:  # w/ vision
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)

        grasp_poses, scores, contact_pts, pc_full, pc_colors = self.grasp_net.infer(mask, rgb, depth * 0.001, self.K_cam_left)

        if visualize:
            self.grasp_net.visualize(rgb, mask, pc_full, grasp_poses, scores, pc_colors)

        all_grasp_poses = []
        all_scores = []
        for obj_id, grasp_poses in grasp_poses.items():
            for i, T_cam_graspPose in enumerate(grasp_poses):
                T_cam_graspPose = normalize_pose(T_cam_graspPose)  # output rotation matrices of network are not normalized

                # set to world frame
                T_world_graspPose = self.T_world_cam @ T_cam_graspPose

                # Set to reachy's gripper frame
                T_world_graspPose = fv_utils.rotateInSelf(T_world_graspPose, [0, 0, 90])
                T_world_graspPose = fv_utils.rotateInSelf(T_world_graspPose, [180, 0, 0])
                T_world_graspPose = fv_utils.translateInSelf(
                    T_world_graspPose, [0, 0, -0.13]
                )  # origin of grasp pose is between fingers for reachy. Value was eyballed
                all_grasp_poses.append(T_world_graspPose)
                all_scores.append(scores[obj_id][i])

        # add the same poses but rotated 180° around z
        for i in range(len(all_grasp_poses)):
            all_grasp_poses.append(fv_utils.rotateInSelf(all_grasp_poses[i], [0, 0, 180]))
            all_scores.append(all_scores[i])

        # Re sorting because we added new grasp poses at the end of the array
        if len(all_grasp_poses) > 0:
            all_scores, all_grasp_poses = zip(*sorted(zip(all_scores, all_grasp_poses), reverse=True, key=lambda x: x[0]))

        print("SCORES: ", all_scores)

        reachable_grasp_poses = []
        reachable_scores = []
        for i, grasp_pose in enumerate(all_grasp_poses):
            # For a grasp pose to be reachable, its pregrasp pose must be reachable too
            # Pregrasp pose is defined as the pose 10cm behind the grasp pose along the z axis of the gripper

            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose = fv_utils.translateInSelf(grasp_pose, [0, 0, 0.1])

            pregrasp_pose_reachable = is_pose_reachable(pregrasp_pose, self.reachy, self.angle_limits, left)

            grasp_pose_reachable = is_pose_reachable(grasp_pose, self.reachy, self.angle_limits, left)

            if pregrasp_pose_reachable and grasp_pose_reachable:
                reachable_grasp_poses.append(grasp_pose)
                reachable_scores.append(all_scores[i])

        print("REACHABLE SCORES: ", reachable_scores)

        return reachable_grasp_poses, reachable_scores

    def execute_grasp(self, grasp_pose: npt.NDArray[np.float32], left: bool = False, duration: float = 2.0) -> bool:

        grasp_pose[:3, 3] += np.array([0, 0, 0.05])  # 0.05 to compensate the gravity

        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose = fv_utils.translateInSelf(pregrasp_pose, [0, 0, 0.1])

        if np.linalg.norm(grasp_pose[:3, 3]) > 1.0:  # safety check
            raise ValueError("Grasp pose is too far away (norm > 1.0)")

        if grasp_pose[:3, 3][0] < 0.0:  # safety check
            raise ValueError("Grasp pose is behind the robot (x < 0)")

        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        joint_pregrasp_pose = arm.inverse_kinematics(pregrasp_pose)
        goto(
            {joint: pos for joint, pos in zip(arm.joints.values(), joint_pregrasp_pose)},
            duration=duration,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        joint_grasp_pose = arm.inverse_kinematics(grasp_pose)
        goto(
            {joint: pos for joint, pos in zip(arm.joints.values(), joint_grasp_pose)},
            duration=duration,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        self.close_gripper(left=left)

        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0, 0, 0.3])
        joint_lift_pose = arm.inverse_kinematics(lift_pose)
        goto(
            {joint: pos for joint, pos in zip(arm.joints.values(), joint_lift_pose)},
            duration=duration,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        return True

    def drop_object(self, target_pose: npt.NDArray[np.float32], left: bool = False) -> None:
        target_pose = np.array(target_pose).reshape(4, 4)
        self.place_object(target_pose, drop_height=0.2)

    def place_object(self, target_pose: npt.NDArray[np.float32], drop_height: float = 0.0, left: bool = False) -> bool:
        target_pose = np.array(target_pose)

        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        target_pose[:3, :3] = self.right_start_pose[:3, :3]
        target_pose[:3, 3] += np.array([0, 0, 0.05 + drop_height])  # 0.05 to compensate the gravity

        # Looking for a reachable target pose
        orig_target_pose = target_pose.copy()
        reachable = is_pose_reachable(target_pose, self.reachy, self.angle_limits, left)
        start = time.time()
        while not reachable:
            if time.time() - start > 2.0:
                print("Timeout : Could not find a reachable target pose. Reverting to default")
                if left:
                    target_pose = fv_utils.rotateInSelf(orig_target_pose, [-45, 0, 0])
                else:
                    target_pose = fv_utils.rotateInSelf(orig_target_pose, [45, 0, 0])
                break

            if left:
                target_pose = fv_utils.rotateInSelf(target_pose, [-0.1, 0, 0])
            else:
                target_pose = fv_utils.rotateInSelf(target_pose, [0.1, 0, 0])
            reachable = is_pose_reachable(target_pose, self.reachy, self.angle_limits, left)

        if reachable:
            print("Found a reachable target pose!")

        joint_target_pose = arm.inverse_kinematics(target_pose)

        goto(
            {joint: pos for joint, pos in zip(arm.joints.values(), joint_target_pose)},
            duration=4.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        self.open_gripper(left=left)

        lift_pose = target_pose.copy()
        lift_pose[:3, 3] += np.array([0, 0, 0.1])
        joint_lift_pose = arm.inverse_kinematics(lift_pose)

        goto(
            {joint: pos for joint, pos in zip(arm.joints.values(), joint_lift_pose)},
            duration=4.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        return True

    def goto_rest_position(self, left: bool = False) -> None:
        if not left:
            goto(
                goal_positions={
                    joint: pos for joint, pos in zip(self.reachy.r_arm.joints.values(), self.right_joint_start_pose)
                },
                duration=4.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
            self.open_gripper(left=False)
        if left:
            goto(
                goal_positions={
                    joint: pos for joint, pos in zip(self.reachy.l_arm.joints.values(), self.left_joint_start_pose)
                },
                duration=4.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
            self.open_gripper(left=True)

    def open_gripper(self, left: bool = False) -> None:
        if left:
            goto({self.reachy.l_arm.l_gripper: 30}, duration=1.0)
        else:
            goto({self.reachy.r_arm.r_gripper: -50}, duration=1.0)

    def close_gripper(self, left: bool = False) -> None:
        if left:
            goto({self.reachy.l_arm.l_gripper: -50}, duration=1.0)
        else:
            goto({self.reachy.r_arm.r_gripper: 20}, duration=1.0)

    def stop(self) -> None:
        print("Stopping the robot...")
        self.reachy.turn_off_smoothly("reachy", duration=3)
        time.sleep(1)
        exit()
