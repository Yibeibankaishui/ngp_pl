import torch
import json
import numpy as np
import os
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, read_image_npy
from .colmap_utils import qvec2rotmat

from .base import BaseDataset


# TODO: implement loading data from droid-slam's result


class DROIDDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)
            
    
    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            # TODO: randomly select depth pixels
            disps = self.disps[img_idxs, pix_idxs]
            depth = self.depth[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                        'rgb': rays[:, :3], 'disps': disps, 'depth': depth}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample
    

    def read_intrinsics(self):
        # ~TODO: read intrinsics
        intrinsics = np.load(os.path.join(self.root_dir, "intrinsics.npy"))
        # fx, fy, cx, cy
        fx = 726
        fy = 726
        # 344 560
        h = int(384*self.downsample)
        w = int(512*self.downsample)
        # fx = intrinsics[0,0]
        # fy = intrinsics[0,1]
        # cx = intrinsics[0,2]
        # cy = intrinsics[0,3]
        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,  1 ]])

        # w = h = int(800*self.downsample)
        # fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

        # K = np.float32([[fx, 0, w/2],
        #                 [0, fy, h/2],
        #                 [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        # TODO: H, W
        # h = int(2 * cy)
        # w = int(2 * cx)
        # ~TODO: what is get_ray_directions
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    # TODO: quaternion to R, t

    def read_meta(self, split):
        # TODO: read meta
        self.rays = []
        self.poses = []
        self.disps = []
        
        # tstamps = np.load(os.path.join(self.root_dir, "tstamps.npy"))
        images = np.load(os.path.join(self.root_dir, "images.npy"))
        disps = np.load(os.path.join(self.root_dir, "disps.npy"))
        poses = np.load(os.path.join(self.root_dir, "poses.npy"))

        print(f'Loading {len(images)} {split} images ...')
        
        # TODO: read poses and rays one by one
        
        #  c2w [R t]
        # pose : [x y z q1 q2 q3 q0]
        q = poses[ : ,3: ]
        # put the real part at the front
        q[...,[0,1,2,3]] = q[...,[3,0,1,2]]
        T = []
        for iq in range(len(q)):
            t = poses[iq, :3].reshape(3,1)
            R = qvec2rotmat(q[iq])
            T += [np.concatenate([R,t],1)]
            
        T = np.stack(T, 0)
        # TODO: shall we use the inv of the pose ?
        c2w = center_poses(T)

        scale = np.linalg.norm(c2w[..., 3], axis=-1).min()

        c2w /= scale
        self.poses = c2w
        # images need to rearrange
        images = rearrange(images, 'N C H W -> N H W C')
        # ~TODO: shift and scale for poses AND images ?
        for img_npy, disp_npy in zip(images, disps):
            # h w c --> (h w) c
            img = read_image_npy(img_npy, self.img_wh, blend_a=False)
            # print(disp_npy.shape)
            # disp_npy = disp_npy[:, :, np.newaxis]
            # print(disp_npy.shape)
            disp = read_image_npy(disp_npy, self.img_wh, blend_a=False)
            self.rays += [img]
            self.disps += [disp]

        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
            # inverse DEPTH
            self.disps = torch.FloatTensor(np.stack(self.disps))
            self.depth = 1/self.disps
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
    