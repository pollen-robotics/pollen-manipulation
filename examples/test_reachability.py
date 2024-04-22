from reachy_sdk import ReachySDK

from pollen_manipulation.reachy.reachability import is_pose_reachable, read_angle_limits

reachy = ReachySDK("172.16.0.32")
angle_limits = read_angle_limits(reachy)

while True:
    pose = reachy.r_arm.forward_kinematics()
    print(is_pose_reachable(pose, reachy, angle_limits, is_left_arm=False))
