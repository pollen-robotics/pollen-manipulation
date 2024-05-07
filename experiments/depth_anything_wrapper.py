
import numpy as np
from transformers import pipeline


class DepthAnythingWrapper:
    def __init__(self):
        # self.checkpoint = "LiheYoung/depth-anything-small-hf" # much faster, don't see much difference in quality
        self.checkpoint = "LiheYoung/depth-anything-large-hf"
        self.pipe = pipeline(task="depth-estimation", model=self.checkpoint)

    def get_depth(self, rgb):
        depth = self.pipe(rgb)["depth"]
        return np.asarray(depth)