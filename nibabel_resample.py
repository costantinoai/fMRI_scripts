# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 00:14:14 2021

@author: 45027900
"""

from nilearn.image import resample_to_img
import nibabel as nib
import glob
import os

masks_root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\masks\v1_v2_v3'
betas_root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\SPM\RSA_blocks_1_lev_6HMP_all'
out_root = os.path.join(masks_root, 'realigned_mni_resampled_spmbetas')
rois = sorted(glob.glob(os.path.join(masks_root, 'realigned_mni', '*')))

interp = "nearest"
# realign_int = "Linear"

for roi in rois:
    sizes = glob.glob(os.path.join(roi, '*'))
    for size in sizes:
        for sub_id in range(2, 26):
            out_dir = os.path.join(
                out_root, os.path.basename(roi), os.path.basename(size))
            os.makedirs(out_dir, exist_ok=True)
            sub = "sub-" + str(sub_id).zfill(2)
            mask = glob.glob(os.path.join(size, sub + '*'))[0]

            filename_in = (os.path.split(mask)[-1])[:-4]
            print(f'STEP: {sub}, {filename_in} -- {mask}')
            src_img = nib.load(mask)
            trgt_img = nib.load(os.path.join(
                betas_root, sub, 'beta_0001.nii'))
            resampled = resample_to_img(
                src_img, trgt_img, interpolation=interp)
            filename_out = filename_in + '_resampled.nii'
            nib.save(resampled, os.path.join(out_dir, filename_out))
