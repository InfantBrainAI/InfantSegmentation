import nibabel as nib
import numpy as np
import os
import pandas as pd
from data.data_util import label_remapping

def cal_dice(pred, tar, tag):
    return (2 * np.sum(np.multiply(pred == tag, tar == tag), axis=None) + 1e-5) / (
                np.sum(pred == tag, axis=None) + np.sum(tar == tag, axis=None) + 1e-5)

ext = '.nii'
gt_path = '/media/win_ssd/Ubuntu/Subcortical_Seg/datasets/CANDI13_duplicate/test_label'
opt_path = '/media/win_ssd/Ubuntu/Subcortical_Seg/datasets/CANDI13_duplicate/results'
tag_num = 7
sbjs = [i for i in os.listdir(gt_path) if i.endswith(ext)]
sbjs = [i.split(ext)[0] for i in sbjs]

mapping_file = '/media/win_ssd/Ubuntu/Subcortical_Seg/Subcortical_mapping.csv'

sbjs = list(set(sbjs))

try:
    nib.load(nib.load(os.path.join(gt_path, sbjs[0] + ext)))
except ValueError:
    nib.Nifti1Header.quaternion_threshold = -1e-06

for i, sbj_id in enumerate(sbjs):
    print(sbj_id)

    GT_scan = np.round(nib.load(os.path.join(gt_path, sbj_id + ext)).get_fdata())
    if mapping_file is not None:
        GT_scan = label_remapping(GT_scan, mapping_file)
    Fake_scan = np.round(nib.load(os.path.join(opt_path, sbj_id + '_7Subcortical_Seg' + ext)).get_fdata())
    print(np.unique(GT_scan))
    DSC = []
    for j in range(1,tag_num+1):
        DSC.append(cal_dice(GT_scan, Fake_scan, j))
    d = {'sbj:': sbj_id}
    for j in range(tag_num):
        d.update({'DSC'+str(j+1): DSC[j]})

    d.update({'meanDSC': np.mean(DSC)})
    if i == 0:
        df = pd.DataFrame(data=d, index=[0])
    else:
        df2 = pd.DataFrame(data=d, index=[i])
        df = df.append(df2)

df.to_excel(os.path.join(opt_path, 'results.xlsx'))

