import argparse
import pickle
import time

import cv2
from pcl_visualizer import PCLVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--data", type=str, default="data.pkl")
args = parser.parse_args()

data = pickle.load(open(args.data, "rb"))

rgb = data["rgb"]
depth = data["depth"]
mono_depth = data["mono_depth"]
K = data["K"]

P1 = PCLVisualizer(data["K"])
P1.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

P2 = PCLVisualizer(data["K"])
P2.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), mono_depth)

while True:
    P1.tick()
    P2.tick()
    time.sleep(0.01)
