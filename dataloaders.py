#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Class of pytorch data loader
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
Aug 10, 2019
"""

import glob
import imageio
import numpy as np
import numpy.matlib
import torch.utils.data
import os
from torchvision import transforms
from config import colorMap
import random
from config import Path
import time
from torch.utils.data import DataLoader


# C_NUM = 12  # number of classes
# 'empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs'
#                0  1  2  3  4   5  6  7  8  9 10  11  12  13  14  15 16 17  18  19  20
seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11,
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11]  # 0 - 11


#                21  22  23  24  25  26  27  28  29 30  31  32 33  34  35  36

def category_based_sample(target, sample_num=10240):
    W,H,D = target.shape
    unit_size = 2 / np.array([W, H, D])
    points_occ, label_occ = [], []
    points_emp, label_emp = [], []
    it = 0
    # calculate it based on sample_num
    num_occupied = len(np.where((target>0)*(target!=255))[0])
    it = int(sample_num/2/num_occupied) + 1
    
    # sample occupied
    for i in range(1, 12):
        index = np.array(np.where(target==i))
        low, high = index.T*unit_size -1, (index.T+1) * unit_size - 1
        P = []
        for _ in range(it):
            p = np.random.uniform(low, high, low.shape)
            P.append(p)
        P = np.row_stack(P)
        lb = np.ones((len(P),1)) * i
        points_occ.append(P)
        label_occ.append(lb)
    points_occ = np.row_stack(points_occ)
    label_occ = np.row_stack(label_occ)
    
    index = random.sample(list(np.arange(len(points_occ))), int(sample_num/2))
    points_occ = points_occ[index]
    label_occ = label_occ[index]
    
    # sample empty
    index = np.array(np.where(target==0))
    low, high = index.T*unit_size -1, (index.T+1) * unit_size - 1
    P = []
    num_empty = len(np.where(target==0)[0])
    it = int(sample_num/2/num_empty) + 1
    for _ in range(it):
        p = np.random.uniform(low, high, low.shape)
        P.append(p)
    points_emp = np.row_stack(P)
    label_emp = np.zeros((len(points_emp),1))
    index = random.sample(list(np.arange(len(points_emp))), int(sample_num/2))
    points_emp = points_emp[index]
    label_emp = label_emp[index]
    
    return np.row_stack([points_occ, points_emp]), np.row_stack([label_occ, label_emp]).reshape(-1)




def in_running_sample(target_hr, sample_num):
    #x_index, y_index, z_index = np.where(target_hr != 255)
    W, H, D = target_hr.shape

    x = np.random.uniform(-1, 1, sample_num)
    y = np.random.uniform(-1, 1, sample_num)
    z = np.random.uniform(-1, 1, sample_num)

    true_x = ((x + 1)/2 * W).astype(np.int)
    true_y = ((y + 1)/2 * H).astype(np.int)
    true_z = ((z + 1)/2 * D).astype(np.int)


    label = target_hr[(true_x, true_y, true_z)]

    return np.array([x, y, z]).transpose((1, 0)), label

def center_sample_points(voxel, voxel_size):
    vx, vy, vz = voxel.shape
    query_points = np.meshgrid(np.arange(0, vx), np.arange(0, vy), np.arange(0, vz))
    x, y, z = query_points
    qx, qy, qz = (x + 0.5) * voxel_size, (y + 0.5) * voxel_size, (z + 0.5) * voxel_size
    label = voxel[(x, y, z)]

    points = np.array([qx, qy, qz]).transpose((1, 2, 3, 0)).reshape((-1, 3))
    label = label.reshape(-1)

    points[:, 0] = 2 * points[:, 0] / (vx * voxel_size) - 1
    points[:, 1] = 2 * points[:, 1] / (vy * voxel_size) - 1
    points[:, 2] = 2 * points[:, 2] / (vz * voxel_size) - 1

    return points, label


def extent_points_per_voxel(points, W, H, D, num_extents=15):
    '''
    randomly sample num_extents points in voxel space for voting
    Args:
        points: [num_point, 3]
        W: width
        H: height
        D: depth
        num_extents: how many points to extent in one voxel,

    Returns:
        final output shape [num_points, num_extents + 1, 3]

    '''
    x_size, y_size, z_size = 2. / W, 2. / H, 2. / D
    all_points = []
    for p in points:
        x, y, z = p
        x_range = (x - x_size / 2, x + x_size / 2)
        y_range = (y - y_size / 2, y + y_size / 2)
        z_range = (z - y_size / 2, z + z_size / 2)
        _x = np.random.uniform(x_range[0], x_range[1], num_extents)
        _y = np.random.uniform(y_range[0], y_range[1], num_extents)
        _z = np.random.uniform(z_range[0], z_range[1], num_extents)
        extent_points = np.array([_x, _y, _z]).transpose((1, 0))
        cat_points = np.row_stack([np.array([p]), extent_points])
        all_points.append(cat_points)
    return np.array(all_points)

class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, root, task="ssc", use_target="hr", batch_pointnum=40960, istest=False):
        self.param = {'voxel_size': (240, 144, 240),
                      'voxel_unit': 0.02,  # 0.02m, length of each grid == 20mm
                      'cam_k': [[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
                                [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
                                [0, 0, 1]],  # fx = K(1,1); fy = K(2,2);
                      }
        #
        self.task = task
        self.batch_pointnum = batch_pointnum
        self.subfix = 'npz'
        self.istest = istest
        self.filepaths = self.get_filelist(root, self.subfix)
        self.use_target = use_target

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] \
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print('Dataset:{} files'.format(len(self.filepaths)))


    def get_item_in_running_sample(self, index, task="ssc"):
        filepath = self.filepaths[index]
        filename = os.path.split(filepath)[1]
        _name = filename[:-4]

        npz_file = np.load(self.filepaths[index])
        try:
            rgb_tensor = npz_file['rgb']
        except:
            rgb_tensor = np.zeros((3,480,640))
        depth_tensor = npz_file['depth']
        if len(depth_tensor.shape) == 2:
            depth_tensor = np.array([depth_tensor])
        try:
            tsdf_tensor = np.array([npz_file['tsdf_hr']])
        except:
            tsdf_tensor = np.array([npz_file['target_hr']])

        target_lr = npz_file['target_lr']
        position = npz_file['position']
        target_hr = npz_file['target_hr']

        if self.use_target == "lr":
            train_points, train_label = category_based_sample(target_lr, self.batch_pointnum)
        elif self.use_target == "hr":
            train_points, train_label = category_based_sample(target_hr, self.batch_pointnum)
        else:
            pass
        test_points, test_label = center_sample_points(target_lr, 0.08)

        if task == "sc":  # only geometry
            train_label[np.where((train_label > 0) * (train_label != 255))] = 1
            test_label[np.where((test_label > 0) * (test_label != 255))] = 1

        if self.istest:
            try:
                tsdf_lr = npz_file['tsdf_lr']  # ( 60,  36,  60)
                nonempty = self.get_nonempty2(tsdf_lr, target_lr, 'TSDF')  # 这个更符合SUNCG的做法
                #nonempty = np.ones(target_lr.shape, dtype=np.float32)
                #nonempty[target_lr == 255] = 0
            except:
                nonempty = np.ones(target_lr.shape, dtype=np.float32)
                nonempty[target_lr == 255] = 0
            #return rgb_tensor, depth_tensor, tsdf_tensor, test_points, test_label, target_lr.transpose(
                #(1, 0, 2)), nonempty, position, _name  # 和采点时使用的yxz坐标系对齐
            return rgb_tensor, depth_tensor, tsdf_tensor, test_points, test_label, target_lr.T, nonempty.T, position, _name  # 和采点时使用的yxz坐标系对齐
        return rgb_tensor, depth_tensor, tsdf_tensor, train_points, train_label, position, _name

    def __getitem__(self, index):
        return self.get_item_in_running_sample(index, self.task)

    def __len__(self):
        return len(self.filepaths)

    def get_filelist(self, root, subfix):
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")
        _filepaths = list()
        if isinstance(root, list):  # 将多个root
            for i, root_i in enumerate(root):
                fp = glob.glob(root_i + '/*.' + subfix)
                fp.sort()
                _filepaths.extend(fp)
        elif isinstance(root, str):
            _filepaths = glob.glob(root + '/*.' + subfix)  # List all files in data folder
            _filepaths.sort()
        if len(_filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))

        return _filepaths

    @staticmethod
    def get_nonempty2(voxels, target, encoding):  # Get none empty from depth voxels
        data = np.ones(voxels.shape, dtype=np.float32)  # init 1 for none empty
        data[target == 255] = 0
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels == 0] = 0
        elif encoding == 'TSDF':
            data[voxels >= np.float32(0.001)] = 0
            data[voxels == 1] = 1

        return data


def make_data_loader(args, **kwargs):
    if args.dataset:
        base_dirs = Path.db_root_dir(args.dataset)

        print('Training data:{}'.format(base_dirs['train']))
        train_loader = DataLoader(
            dataset=NYUDataset(base_dirs['train'],
                               task=args.task,
                               use_target=args.use_target,
                               istest=False,
                               batch_pointnum=args.batch_pointnum),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        print('Validate data:{}'.format(base_dirs['val']))
        val_loader = DataLoader(
            dataset=NYUDataset(base_dirs['val'],
                               task=args.task,
                               use_target=args.use_target,
                               istest=True,
                               batch_pointnum=args.batch_pointnum,),
            batch_size=args.batch_size,  # 1 * torch.cuda.device_count(), 1 for each GPU
            shuffle=False,
            num_workers=args.workers  # 1 * torch.cuda.device_count()
        )

        return train_loader, val_loader


def make_dataloader_for_vis(args, **kwargs):
    if args.dataset:
        base_dirs = Path.db_root_dir(args.dataset)

        print('Training data:{}'.format(base_dirs['train']))
        train_loader = DataLoader(
            dataset=NYUDataset(base_dirs['train'],
                               task=args.task,
                               use_target=args.use_target,
                               istest=True,
                               batch_pointnum=args.batch_pointnum),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )

        print('Validate data:{}'.format(base_dirs['val']))
        val_loader = DataLoader(
            dataset=NYUDataset(base_dirs['val'],
                               task=args.task,
                               use_target=args.use_target,
                               istest=True,
                               batch_pointnum=args.batch_pointnum,),
            batch_size=1,  # 1 * torch.cuda.device_count(), 1 for each GPU
            shuffle=False,
            num_workers=args.workers  # 1 * torch.cuda.device_count()
        )

        return train_loader, val_loader

if __name__ == '__main__':
    # ---- Data loader
    root = "/home/magic/Datasets/nyu-tsdf/NYUCADtest_npz"
    qroot = "/home/magic/Datasets/nyu-tsdf/NYUCADtest_querypoints_npz2"

    # ------------------------------------------------
    data_loader = torch.utils.data.DataLoader(
        dataset=NYUDataset(root, qroot, batch_pointnum=40960,  istest=True),
        batch_size=1,
        shuffle=False,
        num_workers=8
    )
    print(len(data_loader))
    for step, (rgb_tesnor, depth, points, label, position, _filename, _, _) in enumerate(data_loader):
        # print('step:', rgb_tesnor.shape, depth.shape, points.shape, label.shape, position.shape)
        continue
