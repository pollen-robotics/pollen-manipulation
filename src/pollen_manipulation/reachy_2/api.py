import time
from typing import Any, Dict, List, Tuple

import cv2
import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from contact_graspnet_pytorch.wrapper import ContactGraspNetWrapper
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from pollen_manipulation.utils import (
    find_close_reachable_pose,
    get_angle_dist,
    normalize_pose,
)


class Reachy2ManipulationAPI:
    def __init__(
        self,
        reachy: ReachySDK,
        T_world_cam: npt.NDArray[np.float32],
        K_cam_left: npt.NDArray[np.float32],
        simu_preview: bool = True,
    ) -> None:
        self.reachy_real = reachy
        self.reachy = self.reachy_real
        self.simu_preview = simu_preview

        # TODO Maybe remove this if it is too annoying
        if not self.simu_preview:
            inp = input("Warning, simu preview is disabled, continue ? (N/y)")
            if inp.lower() != "y":
                raise ValueError("Cancelling")

        if self.simu_preview:
            if self.reachy._host == "localhost":
                raise ValueError("Simu preview is not available the main robot is localhost")
            self.reachy_simu = ReachySDK("localhost", with_synchro=False)
            if not self.reachy_simu.is_connected():
                raise ValueError("Simu preview is not available, cannot connect to the simu")
            self.reachy_simu.turn_on()  # turn on the simu robot by default
            self.reachy = self.reachy_simu

        self.T_world_cam = T_world_cam
        self.K_cam_left = K_cam_left

        self.right_start_pose = fv_utils.make_pose([0.20, -0.24, -0.23], [0, -90, 0])
        self.left_start_pose = fv_utils.make_pose([0.20, 0.24, -0.23], [0, -90, 0])

        self.grasp_net = ContactGraspNetWrapper()

    def ask_simu_preview(self) -> str:
        """
        If self.simu_preview is True, asks the user if they want to run the move on the simu robot before running it on the real robot.
        Returns True if the user has chosen to run the move on the simu robot, False otherwise.
        """
        ret = "real"
        if not self.simu_preview:
            return ret

        inp = input("Run the move on the simu robot before running it on the real robot ? (Y/n/s[kip]): ")
        if inp.lower() == "n":
            print("Ok, running the move on the real robot")
            ret = "real"
        elif inp.lower() == "s" or inp.lower() == "skip":
            print("Ok, skipping the move")
            ret = "skip"
        else:
            ret = "simu"
        return ret

    def grasp_object(
        self,
        object_info: Dict[str, Any],
        left: bool = False,
        visualize: bool = False,
        grasp_gotos_duration: float = 4.0,
        use_cartesian_interpolation: bool = True,
    ) -> bool:
        pose = object_info["pose"]
        rgb = object_info["rgb"]
        mask = object_info["mask"]
        depth = object_info["depth"]

        if len(pose) == 0:
            return False

        grasp_poses, scores, _, _ = self.get_reachable_grasp_poses(rgb, depth, mask, left=left, visualize=visualize)
        print("===================")
        print("ALL SCORES:")
        print(scores)
        print("=========================")

        if len(grasp_poses) == 0:
            return False

        grasp_pose = grasp_poses[0]
        score = scores[0]
        self.reachy = self.reachy_real
        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        arm.publish_grasp_poses([grasp_pose], [score])
        print("GRASP POSE selected: ", grasp_pose, "SCORE: ", score)

        simu = self.ask_simu_preview()
        while simu == "simu":  # while the user wants to run the move on the simu robot
            grasp_success = self._execute_grasp(
                grasp_pose,
                left=left,
                duration=grasp_gotos_duration,
                use_cartesian_interpolation=use_cartesian_interpolation,
                play_in_simu=True,
            )
            simu = self.ask_simu_preview()

        if simu == "skip":
            return False

        grasp_success = self._execute_grasp(
            grasp_pose, left=left, duration=grasp_gotos_duration, use_cartesian_interpolation=use_cartesian_interpolation
        )

        return grasp_success

    def _get_euler_from_homogeneous_matrix(
        self, homogeneous_matrix: npt.NDArray[np.float32], degrees: bool = False
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        position = homogeneous_matrix[:3, 3]
        rotation_matrix = homogeneous_matrix[:3, :3]
        euler_angles: npt.NDArray[np.float32] = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)
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

    def get_reachable_grasp_poses(  # noqa: C901
        self,
        rgb: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        left: bool = False,
        visualize: bool = False,
    ) -> Tuple[List[npt.NDArray[np.float32]], List[np.float32], List[npt.NDArray[np.float32]], List[np.float32]]:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)

        grasp_poses, scores, contact_pts, openings, pc_full, pc_colors = self.grasp_net.infer(
            mask, rgb, depth * 0.001, self.K_cam_left
        )

        if visualize:
            self.grasp_net.visualize(rgb, mask, pc_full, grasp_poses, scores, pc_colors)

        all_grasp_poses = []
        all_scores = []
        for obj_id, grasp_poses in grasp_poses.items():
            for i, T_cam_graspPose in enumerate(grasp_poses):
                T_cam_graspPose = normalize_pose(T_cam_graspPose)  # output rotation matrices of network are not normalized

                # set the pose from camera frame to world frame
                T_world_graspPose = self.T_world_cam @ T_cam_graspPose

                T_world_graspPose_sym = T_world_graspPose.copy()

                # Set to reachy's gripper frame
                T_world_graspPose = fv_utils.rotateInSelf(T_world_graspPose, [0, 0, 90])
                T_world_graspPose = fv_utils.rotateInSelf(T_world_graspPose, [180, 0, 0])
                # T_world_graspPose = fv_utils.rotateInSelf(T_world_graspPose, [0, -90, 0])

                # as grasp z axis is along the base of the "fork", this distance is 0 with a top grasp (z up and x front)
                dist_top = get_angle_dist(T_world_graspPose[:3, :3], np.eye(3))

                # as grasp z axis is along the base of the "fork", this distance is 0 with a top grasp (z up and x front)
                # front=np.zeros((3,3))
                # front[0][2]=-1
                # front[1][1]=1
                # front[2][0]=1

                if not left:
                    front = np.array(
                        [
                            [0, -0.71, -0.71],
                            [0, 0.71, -0.71],
                            [1, 0, 0],
                        ]
                    )
                else:
                    front = np.array(
                        [
                            [0, 0.71, -0.71],
                            [0, 0.71, 0.71],
                            [1, 0, 0],
                        ]
                    )

                dist_front = get_angle_dist(T_world_graspPose[:3, :3], front)

                orientation_score = 1.0
                # if dist_top != 0.0:
                #     orientation_score /= np.abs(dist_top)
                if np.isnan(dist_front):
                    print(f"NAN dist_front: {T_world_graspPose_sym[:3, :3]}")
                    orientation_score *= 0.000001

                if dist_front == 0.0:
                    print(f"Perfect dist_front?!: {T_world_graspPose_sym[:3, :3]}")

                if dist_front != 0.0 and not np.isnan(dist_front):
                    orientation_score /= 1 + np.abs(dist_front)

                if np.abs(dist_front) > np.pi / 2:
                    orientation_score *= 0.1

                # print(f'Angle dist: {dist} orientation_score: {orientation_score} yaw: {yaw} score: {scores[obj_id][i]}')

                # T_world_graspPose = fv_utils.translateInSelf(
                #     T_world_graspPose, [0, 0, -0.13]
                # )  # origin of grasp pose is between fingers for reachy. Value was eyballed

                T_world_graspPose = fv_utils.translateInSelf(
                    T_world_graspPose, [0, 0, -0.0584]
                )  # Graspnet returns the base of the gripper mesh, we translate to get the base of the opening

                all_grasp_poses.append(T_world_graspPose)
                all_scores.append(orientation_score)

                # rotate 180° along z axis to get symetrical solution
                T_world_graspPose_sym = fv_utils.rotateInSelf(T_world_graspPose_sym, [0, 0, 180])
                # check orientation to score

                T_world_graspPose_sym = fv_utils.rotateInSelf(T_world_graspPose_sym, [0, 0, 90])
                T_world_graspPose_sym = fv_utils.rotateInSelf(T_world_graspPose_sym, [180, 0, 0])

                dist_top = get_angle_dist(T_world_graspPose_sym[:3, :3], np.eye(3))
                dist_front = get_angle_dist(T_world_graspPose_sym[:3, :3], front)
                orientation_score = 1.0
                # if dist_top != 0.0:
                #     orientation_score /= np.abs(dist_top)
                if np.isnan(dist_front):
                    print(f"NAN dist_front sym: {T_world_graspPose_sym[:3, :3]}")
                    orientation_score *= 0.000001

                if dist_front == 0.0:
                    print(f"Perfect dist_front sym?!: {T_world_graspPose_sym[:3, :3]}")

                if dist_front != 0.0 and not np.isnan(dist_front):
                    orientation_score /= 1 + np.abs(dist_front)
                # print(f'Sym Angle dist: {dist} orientation_score: {orientation_score} yaw: {yaw} score: {scores[obj_id][i]}')

                if np.abs(dist_front) > np.pi / 2:
                    orientation_score *= 0.1

                T_world_graspPose_sym = fv_utils.translateInSelf(
                    T_world_graspPose_sym, [0, 0, -0.0584]
                )  # Graspnet returns the base of the gripper mesh, we translate to get the base of the opening

                all_grasp_poses.append(T_world_graspPose_sym)

                all_scores.append(orientation_score)

        # Re sorting because we added new grasp poses at the end of the array
        if len(all_grasp_poses) > 0:
            zipped = zip(all_scores, all_grasp_poses)
            sorted_zipped = sorted(zipped, reverse=True, key=lambda x: x[0])
            all_scores, all_grasp_poses = zip(*sorted_zipped)  # type: ignore

        reachable_grasp_poses = []
        reachable_scores = []
        print(f"Number of grasp poses generated: {len(all_grasp_poses)}")
        for i, grasp_pose in enumerate(all_grasp_poses):
            # For a grasp pose to be reachable, its pregrasp pose must be reachable too
            # Pregrasp pose is defined as the pose 10cm behind the grasp pose along the z axis of the gripper

            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose = fv_utils.translateInSelf(grasp_pose, [0, 0, 0.1])

            lift_pose = grasp_pose.copy()
            lift_pose[:3, 3] += np.array([0, 0, 0.10])  # warning, was 0.20

            pregrasp_pose_reachable = self._is_pose_reachable(pregrasp_pose, left)
            if not pregrasp_pose_reachable:
                print(f"\t pregrasp not reachable")
                continue

            grasp_pose_reachable = self._is_pose_reachable(grasp_pose, left)
            if not grasp_pose_reachable:
                print(f"\t grasp not reachable")
                continue

            lift_pose_reachable = self._is_pose_reachable(lift_pose, left)
            if not lift_pose_reachable:
                print(f"\t lift not reachable")
                continue

            if pregrasp_pose_reachable and grasp_pose_reachable and lift_pose_reachable:
                reachable_grasp_poses.append(grasp_pose)
                reachable_scores.append(all_scores[i])
                # print(f"Grasp pose {i} is reachable")

        print(f"Number of reachable grasp poses: {len(reachable_grasp_poses)}")
        return reachable_grasp_poses, reachable_scores, all_grasp_poses, all_scores

    def synchro_simu_joints(self):
        l_real_joints = self.reachy_real.l_arm.get_joints_positions()
        l_gripper_opening = self.reachy_real.l_arm.gripper.opening

        r_real_joints = self.reachy_real.r_arm.get_joints_positions()
        r_gripper_opening = self.reachy_real.r_arm.gripper.opening

        self.reachy_simu.l_arm.goto_joints(l_real_joints, duration=0.1)
        self.reachy_simu.r_arm.goto_joints(r_real_joints, duration=0.1)

        self.reachy_simu.l_arm.gripper.set_opening(l_gripper_opening)
        self.reachy_simu.r_arm.gripper.set_opening(r_gripper_opening)
        time.sleep(0.2)

    def _execute_grasp(
        self,
        grasp_pose: npt.NDArray[np.float32],
        duration: float,
        left: bool = False,
        use_cartesian_interpolation: bool = True,
        play_in_simu: bool = False,
    ) -> bool:

        if self.simu_preview and play_in_simu:
            self.synchro_simu_joints()
            self.reachy = self.reachy_simu
            # TODO set simu reachy to the same state as the real one
        else:
            self.reachy = self.reachy_real

        print("Executing grasp in ", "simu" if play_in_simu else "real robot")
        grasp_pose = fv_utils.translateInSelf(
            grasp_pose, [0, 0, -0.0584]
        )  # Graspnet returns the base of the gripper mesh, we translate to get the base of the opening
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose = fv_utils.translateInSelf(pregrasp_pose, [0, 0, 0.1])

        if np.linalg.norm(grasp_pose[:3, 3]) > 1.0 or grasp_pose[:3, 3][0] < 0.0:  # safety check
            print("Grasp pose is too far away (norm > 1.0) or x < 0.0")
            return False

        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        self.open_gripper(left=left, play_in_simu=play_in_simu)
        goto_id = arm.goto_from_matrix(
            target=pregrasp_pose, duration=duration, with_cartesian_interpolation=use_cartesian_interpolation
        )

        if goto_id.id == -1:
            print("Goto ID for pregrasp pose is -1")
            return False

        # while not self.reachy.is_move_finished(goto_id):
        #     time.sleep(0.1)

        goto_id = arm.goto_from_matrix(
            target=grasp_pose, duration=duration, with_cartesian_interpolation=use_cartesian_interpolation
        )

        if goto_id.id == -1:
            print("Goto ID for grasp pose is -1")
            return False

        # while not self.reachy.is_move_finished(goto_id):
        #     time.sleep(0.1)

        self.close_gripper(left=left, play_in_simu=play_in_simu)

        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0, 0, 0.10])
        goto_id = arm.goto_from_matrix(
            target=lift_pose, duration=duration, with_cartesian_interpolation=use_cartesian_interpolation
        )
        if goto_id.id == -1:
            print("Goto ID for lift pose is -1")
            return False

        # while not self.reachy.is_move_finished(goto_id):
        #     time.sleep(0.1)

        return True

    # TODO: Implement this method
    def drop_object(self) -> bool:
        return True

    def place_object(
        self,
        target_pose: npt.NDArray[np.float32],
        place_height: float = 0.0,
        duration: float = 4,
        left: bool = False,
        use_cartesian_interpolation: bool = True,
    ) -> bool:

        simu = self.ask_simu_preview()
        while simu == "simu":  # while the user wants to run the move on the simu robot
            place_success = self._place(
                target_pose,
                place_height=place_height,
                duration=duration,
                left=left,
                use_cartesian_interpolation=use_cartesian_interpolation,
                play_in_simu=True,
            )
            simu = self.ask_simu_preview()

        if simu == "skip":
            return False

        place_success = self._place(
            target_pose,
            place_height=place_height,
            duration=duration,
            left=left,
            use_cartesian_interpolation=use_cartesian_interpolation,
        )
        return place_success

    def _place(
        self,
        target_pose: npt.NDArray[np.float32],
        place_height: float = 0.0,
        duration: float = 4,
        left: bool = False,
        use_cartesian_interpolation: bool = True,
        play_in_simu: bool = False,
    ) -> bool:
        """
        Moves the arm to the target pose and then opens the gripper

        Args:
            target_pose (list): 4x4 homogenous matrix representing the target pose
            place_height (float, optional): Height (in meters) to place the object from. (default: 0.0)
            duration (float, optional): Duration of the movement in seconds. (default: 4)
            left (bool, optional): True if the object should be placed with the left arm, False for the right arm. (default: False)
        Returns:
            bool: True if the object was placed successfully, False otherwise

        """

        if self.simu_preview and play_in_simu:
            self.synchro_simu_joints()
            self.reachy = self.reachy_simu
        else:
            self.reachy = self.reachy_real

        print("Executing place in ", "simu" if play_in_simu else "real robot")

        target_pose = np.array(target_pose).reshape(4, 4)  # just in case :)

        # The rotation component of the pose is the identity matrix, but the gripper's frame has a rotation (z aligned with the forearm, x orthogonal to the fingers)
        # We need to rotate it
        if left:
            target_pose[:3, :3] = self.left_start_pose[:3, :3]
        else:
            target_pose[:3, :3] = self.right_start_pose[:3, :3]
        target_pose[:3, 3] += np.array([0, 0, place_height])

        if np.linalg.norm(target_pose[:3, 3]) > 1.0 or target_pose[:3, 3][0] < 0.0:  # safety check
            raise ValueError("Target pose is too far away (norm > 1.0) or x < 0.0")

        if left:
            arm = self.reachy.l_arm
        else:
            arm = self.reachy.r_arm

        # TODO check reachability.
        target_pose = find_close_reachable_pose(target_pose, self._is_pose_reachable, left=left)
        if target_pose is None:
            print("Could not find a reachable target pose.")
            return False

        goto_id = arm.goto_from_matrix(
            target=target_pose, duration=duration, with_cartesian_interpolation=use_cartesian_interpolation
        )

        if goto_id.id == -1:
            print("Goto ID for pregrasp pose is -1")
            return False

        if goto_id.id != 0:
            while not self.reachy.is_move_finished(goto_id):
                print("Waiting for movement to finish...")
                time.sleep(0.1)

        self.open_gripper(left=left, play_in_simu=play_in_simu)

        return True

    def goto_rest_position(
        self,
        left: bool = False,
        open_gripper: bool = True,
        goto_duration: float = 4.0,
        use_cartesian_interpolation: bool = True,
        play_in_simu: bool = False,
    ) -> None:

        if self.simu_preview and play_in_simu:
            self.reachy = self.reachy_simu
        else:
            self.reachy = self.reachy_real

        if not left:
            goto_id = self.reachy.r_arm.goto_from_matrix(
                self.right_start_pose,
                duration=goto_duration,
                with_cartesian_interpolation=use_cartesian_interpolation,
            )
        else:
            goto_id = self.reachy.l_arm.goto_from_matrix(
                self.left_start_pose,
                duration=goto_duration,
                with_cartesian_interpolation=use_cartesian_interpolation,
            )

        if goto_id.id != 0:
            while not self.reachy.is_move_finished(goto_id):
                time.sleep(0.1)

        if open_gripper:
            self.open_gripper(left=left)

    def open_gripper(self, left: bool = False, play_in_simu: bool = False) -> None:
        if self.simu_preview and play_in_simu:
            self.reachy = self.reachy_simu
        else:
            self.reachy = self.reachy_real

        if left:
            self.reachy.l_arm.gripper.open()
        else:
            self.reachy.r_arm.gripper.open()

    def close_gripper(self, left: bool = False, play_in_simu: bool = False) -> None:
        if self.simu_preview and play_in_simu:
            self.reachy = self.reachy_simu
        else:
            self.reachy = self.reachy_real

        if left:
            self.reachy.l_arm.gripper.close()
        else:
            self.reachy.r_arm.gripper.close()

    def turn_robot_on(self) -> None:
        self.reachy = self.reachy_real
        self.reachy.turn_on()

    def stop(self) -> None:
        self.reachy = self.reachy_real
        self.reachy.turn_off_smoothly()
