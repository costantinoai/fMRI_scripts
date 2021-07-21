# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:35:18 2021

@author: 45027900
"""

import os
import glob
import nibabel as nib
import numpy as np

root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\parcels\nii\final_parcels'
sub_list = ['sub-' + str(sub_id).zfill(2) for sub_id in range(2, 26)]

for sub in sub_list:
    path = os.path.join(root, sub)
    ffa = [nib.load(img)
           for img in sorted(glob.glob(os.path.join(path, '*FFA*')))]
    ffa_new = nib.Nifti1Image(ffa[0].get_fdata() + ffa[1].get_fdata(),
                              affine=ffa[0].affine)
    loc = [nib.load(img)
           for img in sorted(glob.glob(os.path.join(path, '*LOC*')))]
    loc_new = nib.Nifti1Image(loc[0].get_fdata() + loc[1].get_fdata(),
                              affine=loc[0].affine)
    nib.save(ffa_new, os.path.join(path, 'FFA.nii'))
    nib.save(loc_new, os.path.join(path, 'LOC.nii'))
    print(f'{sub} done!')
