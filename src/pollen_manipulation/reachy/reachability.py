from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import yaml
from reachy_sdk import ReachySDK
from yaml.loader import BaseLoader


def read_angle_limits(reachy: ReachySDK, path: str = "../src/config_files/") -> Dict:
    # Open and read the arm config files
    right_path = path + "right_arm_advanced.yaml"
    left_path = path + "left_arm_advanced.yaml"

    angle_limits = {}

    with open(right_path) as f:
        data = yaml.load(f, Loader=BaseLoader)
        values = reachy.r_arm.joints.values()
        for j in values:
            offset = 0
            inverted = False
            if "direct" in data["right_arm_advanced"][j.name]["dxl_motor"]:
                inverted = bool(data["right_arm_advanced"][j.name]["dxl_motor"]["direct"])
            if "offset" in data["right_arm_advanced"][j.name]["dxl_motor"]:
                offset = float(data["right_arm_advanced"][j.name]["dxl_motor"]["offset"])
            inverted_coeff = -1.0 if inverted else 1.0
            angle_limits[j.name] = (
                inverted_coeff * float(data["right_arm_advanced"][j.name]["dxl_motor"]["cw_angle_limit"]) - offset,
                inverted_coeff * float(data["right_arm_advanced"][j.name]["dxl_motor"]["ccw_angle_limit"]) - offset,
                inverted,
            )
    with open(left_path) as f:
        data = yaml.load(f, Loader=BaseLoader)
        values = reachy.l_arm.joints.values()
        for j in values:
            offset = 0
            inverted = False
            if "direct" in data["left_arm_advanced"][j.name]["dxl_motor"]:
                inverted = bool(data["left_arm_advanced"][j.name]["dxl_motor"]["direct"])
            if "offset" in data["left_arm_advanced"][j.name]["dxl_motor"]:
                offset = float(data["left_arm_advanced"][j.name]["dxl_motor"]["offset"])

            inverted_coeff = -1.0 if inverted else 1.0
            angle_limits[j.name] = (
                inverted_coeff * float(data["left_arm_advanced"][j.name]["dxl_motor"]["cw_angle_limit"]) - offset,
                inverted_coeff * float(data["left_arm_advanced"][j.name]["dxl_motor"]["ccw_angle_limit"]) - offset,
                inverted,
            )

    return angle_limits


def is_valid_angle(angle: float, angle_limits: List[float]) -> bool:
    """
    angle and angle_limits have to be in the same units
    """
    cw_limit = angle_limits[0]
    ccw_limit = angle_limits[1]
    inverted = angle_limits[2]
    valid = False

    if cw_limit == ccw_limit:
        # Free wheel mode or multi turn mode
        valid = True
    else:
        if not inverted:
            if angle >= cw_limit and angle <= ccw_limit:
                valid = True
        else:
            if angle >= ccw_limit and angle <= cw_limit:
                valid = True

    return valid


def is_valid_angles(joints: Dict, angle_limits: List[float]) -> bool:
    all_ok = True
    for j in joints.keys():
        limits = angle_limits[j.name]
        angle = joints[j] * np.pi / 180.0
        valid = is_valid_angle(angle, limits)
        if not valid:
            all_ok = False
    return all_ok


def is_pose_reachable(
    pose: npt.NDArray[np.float32],
    reachy: ReachySDK,
    angle_limits: List[float],
    is_left_arm: bool = True,
    precision_meters: float = 0.01,
    precision_rads: float = 0.09,
    ik_seed: Optional[List[float]] = None,
) -> bool:
    angles = []
    joints = {}
    # Note: in this version, the same precision is used to compare meters and degrees...
    # Not ideal but shouldn't matter since it's often either almost identical, or not at all
    try:
        if is_left_arm:
            angles = reachy.l_arm.inverse_kinematics(pose, q0=ik_seed)
            real_pose = reachy.l_arm.forward_kinematics(angles)
            joints = {joint: pos for joint, pos in zip(reachy.l_arm.joints.values(), angles)}
        else:
            angles = reachy.r_arm.inverse_kinematics(pose, q0=ik_seed)
            real_pose = reachy.r_arm.forward_kinematics(angles)
            joints = {joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), angles)}
        # Testing if the forward kinematics matches
        for iy, ix in np.ndindex(pose.shape):
            if ix == 3:
                precision = precision_meters
            else:
                precision = precision_rads

            if abs(pose[iy, ix] - real_pose[iy, ix]) > precision:
                return False
        # Testing if the angular limits are respected
        valid = is_valid_angles(joints, angle_limits)
        return valid

    except Exception as e:
        print("Exception in is_pose_reachable : ", e)
        return False


def is_reachable(reachy: ReachySDK) -> bool:

    return True
