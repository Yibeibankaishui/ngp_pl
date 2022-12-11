import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image
from .colmap_utils import qvec2rotmat

from .base import BaseDataset


# TODO: implement l
# oading data from droid-slam's result


class DROIDDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        # TODO: read intrinsics
        intrinsics = np.load(os.path.join(self.root_dir, "intrisics.npy"))
        # fx, fy, cx, cy
        fx = intrinsics[0,0]
        fy = intrinsics[0,1]
        cx = intrinsics[0,2]
        cy = intrinsics[0,3]
        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])

        # w = h = int(800*self.downsample)
        # fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

        # K = np.float32([[fx, 0, w/2],
        #                 [0, fy, h/2],
        #                 [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        h = 2 * cx
        w = 2 * cy
        # ~TODO: what is get_ray_directions
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (h, w)

    # TODO: quaternion to R, t

    def read_meta(self, split):
        # TODO: read meta
        self.rays = []
        self.poses = []
        

        tstamps = np.load(os.path.join(self.root_dir, "tstamps.npy"))
        images = np.load(os.path.join(self.root_dir, "images.npy"))
        disps = np.load(os.path.join(self.root_dir, "disps.npy"))
        poses = np.load(os.path.join(self.root_dir, "poses.npy"))

        print(f'Loading {len(frames)} {split} images ...')
        
        # TODO: read poses and rays one by one
        
        #  c2w [R t]
        # pose : [x y z q1 q2 q3 q0]
        q = poses[ : ,3: ]
        # put the real part at the front
        q[0,1,2,3] = q[3,0,1,2]
        t = poses[ : , :3]
        c2w = qvec2rotmat(q)
        
        
        # images need to rearrange
        rays = rearrange(images, 'N C H W -> N H W C')
        # TODO: shift and scale for poses ?
        

        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
    