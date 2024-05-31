import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict, List, Tuple, Any, Optional

from contact_graspnet_pytorch.wrapper import ContactGraspNetWrapper
import FramesViewer.utils as fv_utils

from reachy2_sdk import ReachySDK
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK

from pollen_manipulation.utils import normalize_pose


class Reachy2ManipulationAPI:
    def __init__(self, reachy: ReachySDK, T_world_cam: npt.NDArray[np.float32], K_cam_left: npt.NDArray[np.float32]):
        self.reachy = reachy

        self.T_world_cam = T_world_cam
        self.K_cam_left = K_cam_left

        self.right_start_pose = fv_utils.make_pose([0.20, -0.24, -0.23], [0, -90, 0])
        self.left_start_pose = fv_utils.make_pose([0.20, 0.24, -0.23], [0, -90, 0])

        self.grasp_net = ContactGraspNetWrapper()

    def grasp_object(
        self, object_info: Dict[str, Any], left: bool = False, visualize: bool = False, grasp_gotos_duration: float = 4.0
    ) -> bool:
        pose = object_info["pose"]
        rgb = object_info["rgb"]
        mask = object_info["mask"]
        depth = object_info["depth"]

        if len(pose) == 0:
            return False

        grasp_pose, _ = self.get_reachable_grasp_poses(rgb, depth, mask, left=left)

        if len(grasp_pose) == 0:
            return False

        print("OBJECT POSE: ", pose)
        print("GRASP POSE: ", grasp_pose[0])

        grasp_success = self.execute_grasp(grasp_pose[0], left=left, duration=grasp_gotos_duration)
        return grasp_success

    def _get_euler_from_homogeneous_matrix(
        self, homogeneous_matrix: npt.NDArray[np.float32], degrees: bool = False
    ) -> List[List[float]]:
        position = homogeneous_matrix[:3, 3]
        rotation_matrix = homogeneous_matrix[:3, :3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)
        return position, euler_angles

    def _is_pose_reachable(self, pose: npt.NDArray[np.float32], left: bool = False) -> bool:
        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        try:
            arm.inverse_kinematics(pose)
        except ValueError:
            return False
        return True

    def get_reachable_grasp_poses(
        self,
        rgb: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        left: bool = False,
        visualize: bool = False,
    ) -> Tuple[List[npt.NDArray[np.float32]], List[np.float32]]:
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
                    T_world_graspPose, [0, 0, -0.0584]
                )  # origin of grasp pose is between fingers for reachy. Value was eyballed
                all_grasp_poses.append(T_world_graspPose)
                all_scores.append(scores[obj_id][i])

        # Re sorting because we added new grasp poses at the end of the array
        if len(all_grasp_poses) > 0:
            zipped = zip(all_scores, all_grasp_poses)
            sorted_zipped = sorted(zipped, reverse=True, key=lambda x: x[0])
            all_scores, all_grasp_poses = zip(*sorted_zipped)  # type: ignore

        reachable_grasp_poses = []
        reachable_scores = []
        print(f"Number of grasp poses: {len(all_grasp_poses)}")
        for i, grasp_pose in enumerate(all_grasp_poses):
            # For a grasp pose to be reachable, its pregrasp pose must be reachable too
            # Pregrasp pose is defined as the pose 10cm behind the grasp pose along the z axis of the gripper

            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose = fv_utils.translateInSelf(grasp_pose, [0, 0, 0.1])

            lift_pose = grasp_pose.copy()
            lift_pose[:3, 3] += np.array([0, 0, 0.20])

            pregrasp_pose_reachable = self._is_pose_reachable(pregrasp_pose, left)
            grasp_pose_reachable = self._is_pose_reachable(grasp_pose, left)
            lift_pose_reachable = self._is_pose_reachable(lift_pose, left)

            if pregrasp_pose_reachable and grasp_pose_reachable and lift_pose_reachable:
                reachable_grasp_poses.append(grasp_pose)
                reachable_scores.append(all_scores[i])

        print(f"Number of reachable grasp poses: {len(reachable_grasp_poses)}")
        return reachable_grasp_poses, reachable_scores

    def execute_grasp(self, grasp_pose: npt.NDArray[np.float32], duration: float, left: bool = False) -> bool:
        print("Executing grasp")
        # print(f"Symbolic value of grasp pose: {self._pose_4x4_to_symbolic_pose(grasp_pose)}")
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose = fv_utils.translateInSelf(pregrasp_pose, [0, 0, 0.1])

        if np.linalg.norm(grasp_pose[:3, 3]) > 1.0 or grasp_pose[:3, 3][0] < 0.0:  # safety check
            raise ValueError("Grasp pose is too far away (norm > 1.0)")

        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        # print("Opening gripper")
        self.open_gripper(left=left)

        # print("Going to pregrasp pose")
        goto_id = arm.goto_from_matrix(target=pregrasp_pose, duration=duration)
        # print("GOTO ID: ", goto_id)

        if goto_id.id == -1:
            return False

        while not self.reachy.is_move_finished(goto_id):
            time.sleep(0.1)

        # return pregrasp_pose, grasp_pose

        print("Going to grasp pose")
        goto_id = arm.goto_from_matrix(target=grasp_pose, duration=duration)

        if goto_id.id == -1:
            print("Goto ID is -1")
            return False

        while not self.reachy.is_move_finished(goto_id):
            time.sleep(0.1)

        # return pregrasp_pose, grasp_pose

        print("Closing gripper")
        self.close_gripper(left=left)

        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0, 0, 0.20])
        goto_id = arm.goto_from_matrix(target=lift_pose, duration=duration)
        if goto_id.id == -1:
            print("Goto ID is -1")
            return False

        while not self.reachy.is_move_finished(goto_id):
            time.sleep(0.1)

        return True

    def drop_object(self) -> bool:
        return True

    def place_object(self) -> bool:
        return True

    def goto_rest_position(self, left: bool = False, open_gripper: bool = True) -> None:
        if not left:
            self.reachy.r_arm.goto_from_matrix(self.right_start_pose, duration=4.0)
        else:
            self.reachy.l_arm.goto_from_matrix(self.left_start_pose, duration=4.0)

        if open_gripper:
            self.open_gripper(left=left)

    def open_gripper(self, left: bool = False) -> None:
        if left:
            self.reachy.l_arm.gripper.open()
        else:
            self.reachy.r_arm.gripper.open()

    def close_gripper(self, left: bool = False) -> None:
        if left:
            self.reachy.l_arm.gripper.close()
        else:
            self.reachy.r_arm.gripper.close()

    def turn_robot_on(self) -> None:
        self.reachy.turn_on()

    def stop(self) -> None:
        self.reachy.turn_off_smoothly()
