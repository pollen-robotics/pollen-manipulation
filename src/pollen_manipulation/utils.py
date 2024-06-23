import time
from typing import Callable, Optional

import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from FramesViewer.viewer import Viewer
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

    if reachable:
        return pose

    print("=== Could not place using the current pose")
    # If the current pose (that uses the current pose) is not reachable, revert to previous strategy

    if not left:
        rot = np.array([[0.0, -0.7071068, -0.7071068], [0.0, 0.7071068, -0.7071068], [1.0, 0.0, 0.0]])
    else:
        rot = np.array([[0.0, 0.7071068, -0.7071068], [0.0, 0.7071068, 0.7071068], [1.0, -0.0, 0.0]])

    pose[:3, :3] = rot

    fv = Viewer()
    fv.start()

    rot_tol = 20  # deg

    for i in range(100):
        thetas = (np.random.rand(2) - 0.5) * 2 * rot_tol
        thetas = np.hstack((thetas, [0]))

        pose_candidate = fv_utils.rotateInSelf(pose.copy(), thetas, degrees=True)
        reachable = reachability_function(pose_candidate, left)
        if reachable:
            return pose
        fv.pushFrame(pose_candidate, "truc")
        time.sleep(0.1)

    # for theta_x in range(0, 40):
    #     # rotate 1 degree around x axis
    #     reachable = reachability_function(pose, left)
    #     if reachable:
    #         return pose
    #     pose = fv_utils.rotateInSelf(pose.copy(), [-1 if left else 1, 0, 0], degrees=True)

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
