import time
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List, Tuple

import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import get_best_discrete_theta

from pollen_manipulation.utils import get_euler_from_homogeneous_matrix


class ParallelGraspPoseReachabilityChecker:
    def __init__(self, nb_processes: int = 10) -> None:
        self.symbolic_ik_solver: Dict[str:SymbolicIK] = {}
        self.orbita3D_max_angle = np.deg2rad(42.5)  # 43.5 is too much
        self.nb_processes = nb_processes
        for arm in ["l_arm", "r_arm"]:
            self.symbolic_ik_solver[arm] = SymbolicIK(
                arm=arm,
                upper_arm_size=0.28,
                forearm_size=0.28,
                gripper_size=0.10,
                wrist_limit=np.rad2deg(self.orbita3D_max_angle),
            )

    def is_reachable(self, pose: npt.NDArray[np.float32], left: bool = False) -> bool:
        name = "l_arm" if left else "r_arm"
        prefered_theta = -4 * np.pi / 6
        if not name.startswith("r"):
            prefered_theta = -np.pi - prefered_theta

        goal_position, goal_orientation = get_euler_from_homogeneous_matrix(pose)
        goal_pose = np.array([goal_position, goal_orientation])

        solver: SymbolicIK = self.symbolic_ik_solver[name]
        # Checks if an interval exists that handles the wrist limits and the elbow limits
        (
            is_reachable,
            interval,
            theta_to_joints_func,
            state_reachable,
        ) = solver.is_reachable(goal_pose)

        if is_reachable:
            # Explores the interval to find a solution with no collision elbow-torso
            is_reachable, theta, state = get_best_discrete_theta(
                None,
                interval,
                theta_to_joints_func,
                20,
                prefered_theta,
                solver.arm,
            )
        else:
            print(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")

        return is_reachable

    def check_n_grasp_poses_reachability(
        self,
        grasp_poses: List[npt.NDArray[np.float32]],
        left: bool = True,
    ) -> List[bool]:
        res = []
        for grasp_pose in grasp_poses:
            res.append(self.check_grasp_pose_reachability(grasp_pose, left))
        return res

    def check_grasp_pose_reachability(
        self,
        grasp_pose: npt.NDArray[np.float32],
        left: bool = True,
    ) -> bool:
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose = fv_utils.translateInSelf(grasp_pose, [0, 0, 0.1])

        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0, 0, 0.10])  # warning, was 0.20

        pregrasp_pose_reachable = self.is_reachable(pregrasp_pose, left)
        # if not pregrasp_pose_reachable:
        #     print(f"\t pregrasp not reachable")
        #     return False

        grasp_pose_reachable = self.is_reachable(grasp_pose, left)
        # if not grasp_pose_reachable:
        #     print(f"\t grasp not reachable")
        #     return False

        lift_pose_reachable = self.is_reachable(lift_pose, left)
        # if not lift_pose_reachable:
        #     print(f"\t lift not reachable")
        #     return False

        if pregrasp_pose_reachable and grasp_pose_reachable and lift_pose_reachable:
            return True

        return False

    def run_parallel(
        self, all_grasp_poses: List[npt.NDArray[np.float32]], all_scores: List[float], left: bool = False
    ) -> Tuple[List[npt.NDArray[np.float32]], List[float]]:
        self.nb_processes = min(self.nb_processes, len(all_grasp_poses))
        chunk_size = len(all_grasp_poses) // self.nb_processes
        poses = [all_grasp_poses[i : i + chunk_size] for i in range(0, len(all_grasp_poses), chunk_size)]
        with Pool(self.nb_processes) as p:
            res = p.starmap(self.check_n_grasp_poses_reachability, zip(poses, repeat(left)))

        # flatten res
        res = [reachable for sublist in res for reachable in sublist]
        reachable_grasp_poses = [pose for pose, reachable in zip(all_grasp_poses, res) if reachable]
        reachable_scores = [score for score, reachable in zip(all_scores, res) if reachable]
        for reachable_pose in reachable_grasp_poses:
            print(reachable_pose)
            print("---")
        print("==============")
        print(res)
        print("==============")
        return reachable_grasp_poses, reachable_scores


# TODO remove from here
def TMP_is_pose_reachable(pose: npt.NDArray[np.float32], reachy: ReachySDK, left: bool = False) -> bool:
    if left:
        arm = reachy.l_arm
    else:
        arm = reachy.r_arm

    try:
        arm.inverse_kinematics(pose)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    nb_poses = 200

    # _poses = np.random.rand(nb_poses, 16)
    # poses = [pose.reshape(4, 4) for pose in _poses]
    left_start_pose = fv_utils.make_pose([0.40, 0.14, -0.3], [0, -90, 0])
    left_start_pose = fv_utils.rotateInSelf(left_start_pose, [-45, 20, 0])
    poses = [left_start_pose for _ in range(nb_poses)]
    scores = np.random.rand(nb_poses)
    print("++ PARALLEL CHECKER ++")
    pgprc = ParallelGraspPoseReachabilityChecker()
    print("checking", nb_poses, "poses with", pgprc.nb_processes, "processes")
    start = time.time()
    pgprc.run_parallel(poses, scores, left=True)
    print("=== took", time.time() - start, "seconds")

    print("++ NORMAL SDK CHECKER ++")
    reachy = ReachySDK("localhost")
    time.sleep(2)
    print("checking", nb_poses, "poses")
    res = []
    start = time.time()
    for pose in poses:
        res.append(TMP_is_pose_reachable(pose, reachy, left=True))
    # print(res)
    print("=== took", time.time() - start, "seconds")
