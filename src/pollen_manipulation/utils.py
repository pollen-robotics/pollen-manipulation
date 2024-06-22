import copy
from itertools import repeat
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import get_best_discrete_theta
from scipy.spatial.transform import Rotation as R


def normalize_pose(pose: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    r = R.from_matrix(pose[:3, :3])
    q = r.as_quat()  # astuce, en convertissant en quaternion il normalise
    r2 = R.from_quat(q)
    good_rot = r2.as_matrix()
    pose[:3, :3] = good_rot

    return pose


def find_close_reachable_pose(
    pose: npt.NDArray[np.float32],
    reachability_function: Callable[[npt.NDArray[np.float32], bool], bool],
    left: bool = False,
) -> Optional[npt.NDArray[np.float32]]:
    """
    Find the closest reachable pose from the given pose.
    If the pose is reachable, return it.
    If no reachable pose is found, return None.

    Parameters:
    - pose: the pose from which to start the search
    - reachability_function: a function that takes a pose and a boolean indicating if the pose is for the left arm.
    - left: a boolean indicating if the pose is for the left arm
    """
    print("Finding close reachable pose ...")
    reachable = reachability_function(pose=pose, left=left)
    print("========= SIMSIMSIM ==========")
    print(pose)
    print("==========")
    if reachable:
        print("FIRST POSE WAS REACHABLE OMG")
        return pose

    for theta_x in range(0, 360):
        # rotate 1 degree around x axis
        reachable = reachability_function(pose, left)
        if reachable:
            return pose
        pose = fv_utils.rotateInSelf(pose, [-1 if left else 1, 0, 0], degrees=True)
        print(theta_x)

    return None


def get_angle_dist(P: npt.NDArray[np.float32], Q: npt.NDArray[np.float32]) -> float:
    """
    Compute the angle distance between two rotation matrices P and Q.
    """
    R = np.dot(P, Q.T)
    # print(f'DEBUG MAT DIST: R {R}')
    cos_theta = (np.trace(R) - 1) / 2
    # print(f'DEBUG MAT DIST: cos_theta {cos_theta}')
    angle_dist: float = np.arccos(cos_theta)  # * (180/np.pi)
    # print(f'DEBUG MAT DIST: angle_dist {angle_dist}')
    return angle_dist


def get_euler_from_homogeneous_matrix(
    homogeneous_matrix: npt.NDArray[np.float32], degrees: bool = False
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    position = homogeneous_matrix[:3, 3]
    rotation_matrix = homogeneous_matrix[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)
    return position, euler_angles


# def symbolic_ik_is_reachable(pose: npt.NDArray[np.float64], name: str, symbolic_ik_solver) -> bool:  # type: ignore
#     prefered_theta = -4 * np.pi / 6
#     if not name.startswith("r"):
#         prefered_theta = -np.pi - prefered_theta

#     goal_position, goal_orientation = get_euler_from_homogeneous_matrix(pose)
#     goal_pose = np.array([goal_position, goal_orientation])

#     solver: SymbolicIK = symbolic_ik_solver[name]
#     # Checks if an interval exists that handles the wrist limits and the elbow limits
#     (
#         is_reachable,
#         interval,
#         theta_to_joints_func,
#         state_reachable,
#     ) = solver.is_reachable(goal_pose)

#     if is_reachable:
#         # Explores the interval to find a solution with no collision elbow-torso
#         is_reachable, theta, state = get_best_discrete_theta(
#             None,
#             interval,
#             theta_to_joints_func,
#             20,
#             prefered_theta,
#             solver.arm,
#         )
#     else:
#         print(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")

#     return is_reachable


# def check_grasp_pose_reachability(
#     grasp_pose: npt.NDArray[np.float32],
#     reachability_function: Callable[[npt.NDArray[np.float32], bool], bool],
#     left: bool = True,
# ) -> bool:
#     pregrasp_pose = grasp_pose.copy()
#     pregrasp_pose = fv_utils.translateInSelf(grasp_pose, [0, 0, 0.1])

#     lift_pose = grasp_pose.copy()
#     lift_pose[:3, 3] += np.array([0, 0, 0.10])  # warning, was 0.20

#     pregrasp_pose_reachable = reachability_function(pregrasp_pose, left)
#     if not pregrasp_pose_reachable:
#         print(f"\t pregrasp not reachable")
#         return False

#     grasp_pose_reachable = reachability_function(grasp_pose, left)
#     if not grasp_pose_reachable:
#         print(f"\t grasp not reachable")
#         return False

#     lift_pose_reachable = reachability_function(lift_pose, left)
#     if not lift_pose_reachable:
#         print(f"\t lift not reachable")
#         return False
#     return True


def turbo_parallel_grasp_poses_reachability_check(
    all_grasp_poses: List[npt.NDArray[np.float32]],
    all_scores: List[float],
    reachability_function: Callable[[npt.NDArray[np.float32], bool], bool],
    left: bool = False,
) -> Tuple[List[npt.NDArray[np.float32]], List[float], List[npt.NDArray[np.float32]], List[float]]:

    with Pool(10) as p:
        res = p.starmap(check_grasp_pose_reachability, zip(all_grasp_poses, repeat(reachability_function), repeat(left)))
    print("======")
    print(res)
    print("======")
    exit()

    return [], [], [], []
    return reachable_grasp_poses, reachable_scores, all_grasp_poses, all_scores
