import pickle
import time

import cv2
import FramesViewer.utils as fv_utils
import numpy as np
from depth_anything_wrapper import DepthAnythingWrapper
from pcl_visualizer import PCLVisualizer
from PIL import Image
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path

T_world_cam = fv_utils.make_pose([0.03, -0.15, 0.1], [0, 0, 0])
T_world_cam[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_world_cam = fv_utils.rotateInSelf(T_world_cam, [-45, 0, 0])

cam = SDKWrapper(get_config_file_path("CONFIG_SR"), compute_depth=True)

depth_anything = DepthAnythingWrapper()
pcl_vis = PCLVisualizer(cam.get_K())

waiting = 2  # seconds
start = time.time()
print("Press any key to compute mono depth")
while True:
    data, _, _ = cam.get_data()
    if time.time() - start < waiting:
        continue

    rgb = data["left"]
    cv2.imshow("rgb", rgb)
    key = cv2.waitKey(1)
    depth = data["depth"]
    # enter
    if key == 13:
        print("Computing mono depth")
        mono_depth = depth_anything.get_depth(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)))
        pcl_vis.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), mono_depth)
        cv2.imshow("mono_depth", mono_depth)

        print("Saving data.pkl")
        data = {
            "rgb": rgb,
            "depth": depth,
            "mono_depth": mono_depth,
            "K": cam.get_K()
        }
        pickle.dump(data, open("data.pkl", "wb"))

    pcl_vis.tick()


