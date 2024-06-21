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
    reachable = reachability_function(pose, left)
    if reachable:
        return pose

    # for theta_x in range(0, 180):
    for theta_x in range(0, 360):

        # rotate 1 degree around x axis
        reachable = reachability_function(pose, left)
        if reachable:
            return pose
        pose = fv_utils.rotateInSelf(pose, [-1 if left else 1, 0, 0], degrees=True)

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
