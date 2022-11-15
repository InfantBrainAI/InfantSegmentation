import os
import json
import logging
from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
import numpy as np
import torch
from data.data_util import norm_img, patch_slicer, get_bounds
import boto3
import tempfile
import torchio as tio


def create_temp_filename(object_key):
    suffix = "_" + os.path.basename(object_key)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    return f.name


def model_fn(model_dir):
    """
    Load the model for inference
    """
    logging.info("Invoking user-defined model_fn")
    print("model_fn")

    opt = TestOptions().parse(save=False)
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    G = create_model(opt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G.to(device)
    G.eval()
    G.load_state_dict(torch.load("/opt/ml/model/code/models/ckpt/FINNEAS_V1.pth"))

    model_dict = {'model': G, 'opt': opt}

    return model_dict


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    print("predict_fn")

    tmp_scans, s3_output, ornt_flag, ornt_trans, ori_affine, ori_header = input_data

    G = model['model']
    opt = model['opt']
    # define matrix to store prediction and normalization matrices
    pred = np.zeros((opt.cls_num,) + tmp_scans.shape)
    tmp_norm = np.zeros((opt.cls_num,) + tmp_scans.shape)
    # normalize image
    if opt.normalize:
        tmp_scans = norm_img(tmp_scans, opt.norm_perc)
    scan_patches, tmp_path, tmp_idx = patch_slicer(tmp_scans, tmp_scans, opt.patch_size, opt.patch_stride,
                                                   remove_bg=True, test=True, ori_path=None)
    print(len(scan_patches))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bound = get_bounds(torch.from_numpy(tmp_scans))
    global_scan = torch.unsqueeze(torch.from_numpy(tmp_scans).to(dtype=torch.float), dim=0)

    sbj = tio.Subject(
        one_image=tio.ScalarImage(tensor=global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]))
    transforms = tio.transforms.Resize(target_shape=(80, 80, 80))
    sbj = transforms(sbj)
    down_scan = sbj['one_image'].data

    # go through all patches
    with torch.no_grad():
        for idx, patch in enumerate(scan_patches):
            ipt = torch.from_numpy(patch).to(dtype=torch.float).to(device)
            ipt = ipt.reshape((1, 1,) + ipt.shape)

            tmp_pred = G(ipt, None, down_scan.cuda().reshape([1, 1, 80, 80, 80]), None, None, None,
                         mask_ratio=0, pseudo=True)

            patch_idx = tmp_idx[idx]
            patch_idx = (slice(0, opt.cls_num),) + (
                slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
            pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
            tmp_norm[patch_idx] += 1

    pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
    sf = torch.nn.Softmax(dim=0)
    pred_vol = sf(torch.from_numpy(pred)).numpy()
    pred_vol = np.argmax(pred_vol, axis=0).astype(np.int)
    if ornt_flag == -1:
        pred_vol = nib.apply_orientation(pred_vol, ornt_trans).astype(np.int)
    sav_img = nib.Nifti1Image(pred_vol, ori_affine, header=ori_header)
    print("Saving output file")
    output_filename = create_temp_filename(s3_output["object"])
    print(output_filename)
    nib.save(sav_img, output_filename)
    print("Save output complete")

    print("returning Done")
    return s3_output, output_filename


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    print("request_content_type=", request_content_type)
    if request_content_type == "application/json":
        request = json.loads(request_body)
        print(request)
        s3 = boto3.client('s3')
        print("Writing input file from s3")
        s3_input = request["input"]
        s3_output = request["output"]
        temp_file_path = create_temp_filename(s3_input["object"])
        print(temp_file_path)
        s3.download_file(s3_input['bucket'], s3_input['object'], temp_file_path)
        request['input_file']: temp_file_path
        try:
            nib.load(temp_file_path)
        except ValueError:
            nib.Nifti1Header.quaternion_threshold = -1e-06
        ori_scan = nib.load(temp_file_path)

        ornt_flag = 0
        tar_orientation = ('R', 'A', 'S')
        if nib.aff2axcodes(ori_scan.affine) == tar_orientation:
            tmp_scans = ori_scan.get_fdata()
            ornt_trans = None
        else:
            ori_scnas_new_ortn = nib.as_closest_canonical(ori_scan)
            ornt_trans = nib.orientations.ornt_transform(nib.io_orientation(ori_scnas_new_ortn.affine),
                                                         nib.io_orientation(ori_scan.affine))
            tmp_scans = ori_scnas_new_ortn.get_fdata()
            ornt_flag = -1

        tmp_scans = np.squeeze(tmp_scans)
        tmp_scans[tmp_scans < 0] = 0
        print("Returning tmp_scans, s3_output")
        os.unlink(temp_file_path)
        return tmp_scans, s3_output, ornt_flag, ornt_trans, ori_scan.affine, ori_scan.header

    else:
        raise ValueError("Unsupported content type" + request_content_type)


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    print("output_fn request_content_type=", response_content_type)
    if response_content_type == "application/json":
        s3_output, file_path = prediction

        s3 = boto3.client('s3')
        temp_file_path = file_path
        s3.upload_file(temp_file_path, s3_output['bucket'], s3_output['object'])
        os.unlink(temp_file_path)
        return str(s3_output)
    else:
        raise ValueError("Unsupported content type" + request_content_type)
