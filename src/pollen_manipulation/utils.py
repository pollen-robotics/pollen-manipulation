from typing import Callable, Optional

import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
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
    pos_tol: float = 0.0,  # meters
    rot_tol: int = 0,  # degrees
) -> Optional[npt.NDArray[np.float32]]:
    """
    Find the closest reachable pose from the given pose.
    If the pose is reachable, return it.
    If no reachable pose is found, return None.

    Parameters:
    - pose: the pose from which to start the search
    - reachability_function: a function that takes a pose and a boolean indicating if the pose is for the left arm.
    - left: a boolean indicating if the pose is for the left arm
    - pos_tol: positional tolerance in meters
    - rot_tol: rotational tolerance in degrees
    """
    print("Finding close reachable pose ...")
    reachable = reachability_function(pose, left)
    if reachable:
        return pose

    for theta_x in range(0, 180):
        # Checking all combinations of translations and rotations between tolerances
        # this is WAY too long to compute even with small tolerances
        for y in range(int(-pos_tol * 100), int(pos_tol * 100), 1):
            for z in range(int(-pos_tol * 100), int(pos_tol * 100), 1):
                for theta_y in range(-rot_tol, rot_tol, 1):
                    for theta_z in range(-rot_tol, rot_tol, 1):
                        candidate_pose: npt.NDArray[np.float32] = fv_utils.translateInSelf(pose.copy(), [0, y / 100, z / 100])
                        candidate_pose = fv_utils.rotateInSelf(candidate_pose, [0, 0, theta_z], degrees=True)
                        candidate_pose = fv_utils.rotateInSelf(candidate_pose, [0, theta_y, 0], degrees=True)
                        reachable = reachability_function(candidate_pose, left)
                        if reachable:
                            return candidate_pose
                        else:
                            print("(", theta_x, y, z, theta_y, theta_z, ") Not reachable")

        # rotate 1 degree around x axis
        pose = fv_utils.rotateInSelf(pose, [-1 if left else 1, 0, 0], degrees=True)
        reachable = reachability_function(pose, left)
        if reachable:
            return pose

    return None


def get_angle_dist(P: npt.NDArray[np.float32], Q: npt.NDArray[np.float32]) -> float:
    """
    Compute the angle distance between two rotation matrices P and Q.
    """
    R = np.dot(P, Q.T)
    print(f'DEBUG MAT DIST: R {R}')
    cos_theta = (np.trace(R) - 1) / 2
    print(f'DEBUG MAT DIST: cos_theta {cos_theta}')
    angle_dist: float = np.arccos(cos_theta)  # * (180/np.pi)
    print(f'DEBUG MAT DIST: angle_dist {angle_dist}')
    return angle_dist
