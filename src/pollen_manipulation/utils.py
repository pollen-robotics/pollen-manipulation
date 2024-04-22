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
