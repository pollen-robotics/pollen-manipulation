import numpy as np
from contact_graspnet_pytorch.wrapper import ContactGraspNetWrapper

c = ContactGraspNetWrapper()

data = np.load("/home/antoine/Téléchargements/0.npy", allow_pickle=True).item()
grasp_poses, _, _ = c.infer(data["seg"], data["rgb"], data["depth"], data["K"])
