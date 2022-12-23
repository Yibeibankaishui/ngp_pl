# import numpy as np
# import os
import torch
import time
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio
import cv2 as cv
# import matplotlib.pyplot as plt

root_dir = '/root/autodl-tmp/DROID-SLAM_reconstructions_01/test_01'
tstamps = np.load(os.path.join(root_dir, "tstamps.npy"))
images = np.load(os.path.join(root_dir, "images.npy"))
disps = np.load(os.path.join(root_dir, "disps.npy"))
poses = np.load(os.path.join(root_dir, "poses.npy"))
intrinsics = np.load(os.path.join(root_dir, "intrinsics.npy"))

print(images.shape)
print(disps.shape)

print(images)