#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:26:32 2021

@author: andrea
"""
import glob
import os
from nipype.interfaces.ants import ApplyTransforms
from nilearn import image, plotting

## RETURN LIST OF non-empty (SUB)FOLDERS IN root
def get_results_folders(root):
    
    def check_for_files(filepath):
        for filepath_object in glob.glob(filepath):
            try:
                accept = os.path.isfile(glob.glob(os.path.join(filepath_object, '*.*'))[0])
            except:
                accept = False
        return accept, filepath_object
    
    def fast_scandir(dirname):
        subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            subfolders.extend(fast_scandir(dirname))
        return subfolders
    
    folders = fast_scandir(root)
    res_folders = []
    
    for folder in folders:
        acc_bool, path = check_for_files(folder)
        if acc_bool is True:
            res_folders.append(path)
        else:
            pass
        
    return res_folders

masks_root = r'/mnt/c/Users/45027900/Desktop/original'
trgt_root = r'/mnt/c/Users/45027900/Desktop/resample/T1w_MNI/'
reg_root = r'/mnt/c/Users/45027900/Desktop/resample/h5/'
out_root = r'/mnt/c/Users/45027900/Desktop/original_realigned'
interps = ['Linear', 'NearestNeighbor', 'BSpline']

mask_dirs = get_results_folders(masks_root)

for mask_dir in mask_dirs:
    masks = glob.glob(os.path.join(mask_dir, 'sub*.nii.gz'))
    size = mask_dir.split('/')[-1]
    for interp in interps:
        print(f'##### START: {os.path.join(mask_dir.split("/")[-2], mask_dir.split("/")[-1])} - {interp} #####')
        out_dir = os.path.join(out_root, mask_dir.split('/')[-2], mask_dir.split('/')[-1], interp)
        try:
            os.makedirs(out_dir)
        except:
            pass
        for mask in masks:
            filename = (os.path.basename(os.path.normpath(mask))).split('.')[0]
            if not os.path.exists(os.path.join(out_dir, filename + '_realignedT1MNI.nii')):
                sub = (os.path.basename(os.path.normpath(mask))).split('_')[0]
                ref_img = glob.glob(trgt_root + sub + '*')[0]
                transform_file = glob.glob(reg_root + sub + '*')[0]
                print(f'STEP: Running {filename} - Interpolation {interp}, size {size}')
                # transform
                at = ApplyTransforms()
                at.inputs.input_image = mask
                at.inputs.reference_image = ref_img
                at.inputs.output_image = os.path.join(out_dir, filename + '_realignedT1MNI.nii')
                at.inputs.transforms = transform_file
                at.inputs.interpolation =  interp
                at.inputs.num_threads = 12
                at.inputs.float = True
                at.cmdline
                at.run()
            else:
                pass
        
