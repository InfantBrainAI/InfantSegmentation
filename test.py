import os
from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
import numpy as np
import torch
from data.data_util import norm_img, patch_slicer

opt = TestOptions().parse(save=False)
opt.nThreads = 0  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

G = create_model(opt)
print(G)
G.cuda()
G.eval()

G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, opt.whichmodel)))

test_path = os.path.join(opt.dataroot, 'test_img')

des = os.path.join(opt.dataroot, 'results')
if not os.path.exists(des):
    os.mkdir(des)

test_lst = [i for i in os.listdir(test_path) if i.endswith(opt.extension)]


with torch.no_grad():
    for i in test_lst:
        # load test scans, define output name
        print('Loading test image: ' + i)
        pred_name = i.split(opt.extension)[0] + '_7Subcortical_Seg' + opt.extension
        try:
            nib.load(os.path.join(test_path,i))
        except ValueError:
            nib.Nifti1Header.quaternion_threshold = -1e-06
        tmp_scans = np.squeeze(nib.load(os.path.join(test_path,i)).get_fdata())
        tmp_scans[tmp_scans < 0] = 0

        # define matrix to store prediction and normalization matrices
        pred = np.zeros((opt.cls_num,)+tmp_scans.shape)
        tmp_norm = np.zeros((opt.cls_num,) + tmp_scans.shape)
        # normalize image
        if opt.normalize:
            tmp_scans = norm_img(tmp_scans, opt.norm_perc)
        scan_patches, tmp_path, tmp_idx = patch_slicer(tmp_scans, tmp_scans, opt.patch_size, opt.patch_stride,
                                                       remove_bg=True, test=True, ori_path=None)
        # go through all patches
        for idx, patch in enumerate(scan_patches):
            ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
            tmp_pred = G(ipt.reshape((1,1,)+ipt.shape))
            patch_idx = tmp_idx[idx]
            patch_idx = (slice(0, opt.cls_num),) + (slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
            pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
            tmp_norm[patch_idx] += 1

        pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
        sf = torch.nn.Softmax(dim=0)
        pred_vol = sf(torch.from_numpy(pred)).numpy()
        pred_vol = np.argmax(pred_vol, axis=0).astype(np.float)
        ori_scan = nib.load(os.path.join(test_path, i))
        sav_img = nib.Nifti1Image(np.round(pred_vol).astype(np.int), ori_scan.affine, header=ori_scan.header)
        nib.save(sav_img, os.path.join(des, pred_name))
