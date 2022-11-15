### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### This script was modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from data.base_dataset import BaseDataset
import torch
import nibabel as nib
import torchio as tio
from .data_util import label_remapping, norm_img, make_dataset, patch_slicer
import random
import numpy as np


# Data naming:
# TODO: for subject1, the data structure for training should be as follow:
# train_img ...
#       subject1_T1.nii (.nii.gz)
#       subject1_T2.nii (.nii.gz)
# train_label
#       subject1.nii (.nii.gz)
# for inference, you don't need to provide the label folder


# use RAMDataset if the dataset is not too large to be feed into RAM

class RAMDataset(BaseDataset):
    def initialize(self, opt, phase):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = phase

        ### input A (scan image)
        dir_A = '_img'
        self.dir_A = os.path.join(opt.dataroot, self.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.extension))

        ### input B (real label)
        if opt.isTrain:
            dir_B = '_label'
            self.dir_B = os.path.join(opt.dataroot, self.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B, opt.extension))

        self.img_patch = []
        self.label_patch = []

        '''
        If encounter Error
        '''
        try:
            nib.load(self.A_paths[0])
        except ValueError:
            nib.Nifti1Header.quaternion_threshold = -1e-06

        # get  weight for different tags
        self.weight = torch.FloatTensor([0 for _ in range(opt.cls_num)])

        '''
        Initialize for the training / validation
        '''
        if self.phase == 'train':
            for i, j in enumerate(self.A_paths):
                print('Loading ' + phase + ' image: ' + j)
                tmp_scans = np.squeeze(nib.load(j).get_fdata())
                tmp_scans[tmp_scans < 0] = 0
                print('Loading ' + phase + ' label: ' + self.B_paths[i])
                tmp_label = np.squeeze(np.round(nib.load(self.B_paths[i]).get_fdata()))
                assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'
                if opt.remapping:
                    tmp_label = label_remapping(tmp_label, opt.remap_csv)
                print('num of tags ', len(np.unique(tmp_label)))
                if opt.normalize:
                    tmp_scans = norm_img(tmp_scans, opt.norm_perc)

                self.weight += torch.FloatTensor([(tmp_label == tmp_idx).sum() for tmp_idx in range(opt.cls_num)])
                scan_patches, mask_patches = patch_slicer(tmp_scans, tmp_label, opt.patch_size, opt.patch_stride,
                                                          opt.remove_bg)
                self.img_patch += scan_patches
                self.label_patch += mask_patches
        elif self.phase == 'val':
            for i, j in enumerate(self.A_paths):
                print('Loading ' + phase + ' image: ' + j)
                tmp_scans = np.squeeze(nib.load(j).get_fdata())
                tmp_scans[tmp_scans < 0] = 0
                print('Loading ' + phase + ' label: ' + self.B_paths[i])
                tmp_label = np.squeeze(np.round(nib.load(self.B_paths[i]).get_fdata()))
                assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'
                if opt.remapping:
                    tmp_label = label_remapping(tmp_label, opt.remap_csv)
                print('num of tags ', len(np.unique(tmp_label)))

                if opt.normalize:
                    tmp_scans = norm_img(tmp_scans, opt.norm_perc)

                self.weight += torch.FloatTensor([(tmp_label == tmp_idx).sum() for tmp_idx in range(opt.cls_num)])
                scan_patches, mask_patches = patch_slicer(tmp_scans, tmp_label, opt.patch_size, opt.patch_size,
                                                          opt.remove_bg)
                self.img_patch += scan_patches
                self.label_patch += mask_patches
        else:
            self.patch_idx = []
            self.patch_path = []
            for i, j in enumerate(self.A_paths):
                print('Loading ' + self.phase + ' image: ' + j)
                tmp_scans = np.squeeze(nib.load(j).get_fdata())
                tmp_scans[tmp_scans < 0] = 0

                if opt.normalize:
                    tmp_scans = norm_img(tmp_scans, opt.norm_perc)

                scan_patches, tmp_path, tmp_idx = patch_slicer(tmp_scans, tmp_scans, opt.patch_size, opt.patch_stride,
                                                               opt.remove_bg, test=True, ori_path=j)
                self.img_patch += scan_patches
                self.patch_idx += tmp_idx
                self.patch_path += tmp_path

        self.dataset_size = len(self.img_patch)
        self.weight = 1 / self.weight
        self.weight = torch.nan_to_num(self.weight, posinf=0)
        self.weight /= self.weight.sum()

    def __getitem__(self, index):

        '''
        getitem for training/validation
        '''

        if self.phase != 'test':

            A_tensor = torch.from_numpy(self.img_patch[index]).to(dtype=torch.float)
            if len(A_tensor.shape) == 3:
                A_tensor = torch.unsqueeze(A_tensor, 0)

            B_tensor = torch.from_numpy(self.label_patch[index]).long()
            if len(B_tensor.shape) == 3:
                B_tensor = torch.unsqueeze(B_tensor, 0)
            if self.opt.aug and self.phase == 'train':
                tmp = random.uniform(0, 1)
                if tmp > self.opt.aug_prob:
                    A_tensor *= random.uniform(0.95, 1.05)

            A_tensor = tio.ScalarImage(tensor=A_tensor)
            B_tensor = tio.LabelMap(tensor=B_tensor)

            transforms = (tio.RandomAffine(scales=(0.6, 1.8), degrees=20,
                                           isotropic=False, translation=20,
                                           default_pad_value=0, image_interpolation='linear',
                                           label_interpolation='nearest'),
                          tio.RandomFlip(axes=['LR', 'AP', 'IS'], flip_probability=0.5),
                          tio.OneOf([tio.RandomNoise(mean=0.5, std=0.01, p=0.25), tio.RandomBiasField(p=0.25)]))
            transforms = tio.Compose(transforms)

            if self.opt.aug and self.phase == 'train':
                sbj = tio.Subject(one_image=A_tensor, a_segmentation=B_tensor)
                tmp = random.uniform(0, 1)
                if tmp > self.opt.aug_prob:
                    sbj = transforms(sbj)
                    A_tensor = sbj['one_image']
                    B_tensor = sbj['a_segmentation']
            input_dict = {'img': A_tensor.data, 'label': torch.squeeze(B_tensor.data.long())}

            return input_dict
        else:
            A_tensor = torch.from_numpy(self.img_patch[index]).to(dtype=torch.float)
            if len(A_tensor.shape) == 3:
                A_tensor = torch.unsqueeze(A_tensor, 0)
            input_dict = {'img': A_tensor, 'path': self.patch_path[index], 'idx': self.patch_idx[index]}
            return input_dict

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'AlignedDataset'

    def weight(self):
        return self.weight

# class RegularDataset(BaseDataset):
#     def initialize(self, opt):
#         self.opt = opt
#         self.root = opt.dataroot
#
#         ### input A (label maps)
#         dir_A = '_A' if self.opt.label_nc == 0 else '_label'
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
#         self.A_paths = sorted(make_dataset(self.dir_A))
#         ### input B (real images)
#         # if opt.isTrain:
#         dir_B = '_B' if self.opt.label_nc == 0 else '_img'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
#         self.B_paths = sorted(make_dataset(self.dir_B))
#
#         self.dataset_size = len(self.A_paths)
#         # transforms = (
#         #     tio.RandomAffine(scales=(0.8,1.2), degrees=25,isotropic=True,translation=10, default_pad_value=0,image_interpolation=),
#         #     tio.RandomElasticDeformation
#         #     tio.RandomFlip(axes=['LR', 'AP', 'IS']),
#         #     tio.OneOf([tio.RandomAnisotropy(), tio.RandomElasticDeformation()]),
#         # )
#         # self.transform = tio.Compose(transforms)
#
#     def __getitem__(self, index):
#         ### input A (label maps)
#         A_path = self.A_paths[index]
#         A = nib.load(A_path).get_data()
#         A_tensor = torch.from_numpy(A).to(dtype=torch.float)
#         A_tensor = torch.unsqueeze(A_tensor, 0)
#         # if len(A_tensor.shape) == 2:
#
#
#         ### input B (real images)
#         # if self.opt.isTrain:
#         B_path = self.B_paths[index]
#         B = nib.load(B_path).get_data()
#         B_tensor = torch.from_numpy(B).long()
#         # B_tensor = tio.LabelMap(tensor=
#             # if len(B_tensor.shape) == 2:
#         # B_tensor = torch.unsqueeze(B_tensor, 0)
#
#         input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}
#         return input_dict
#
#     def __len__(self):
#         return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize
#
#     def name(self):
#         return 'AlignedDataset'
