# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:35:00 2021

@author: 45027900
"""

import neuropythy as ny
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

ny.config['freesurfer_subject_paths'] =  r'D:\Andrea\fov\bids\derivatives\freesurfer'

root = os.path.abspath(r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS')
freesurfer_path = os.path.join(root, 'derivatives', 'freesurfer')
out_root = os.path.join(root, 'derivatives', 'masks','new')
subjects = sorted(os.listdir(freesurfer_path))

## Parameters
fov_radius_list = np.arange(0.5, 4.1, 0.5)
per_radius_list = np.arange(0.5, 1.51, 0.25)

per_ecc = 7
# stim1_normal_ang = 225 # 3rd quadrant
# stim2_normal_ang = 45 # 1st quadrant
# stim1_inverted_ang = 135 # 2nd qudrant
# stim2_inverted_ang = 315 # 4th quadrant

def get_retinotopic(subject, session):
    try:
        # Look for precomputed sessions
        with open(session, 'rb') as f:  
            data = pickle.load(f)
        lh_visual = {'visual_area': data[0], 'polar_angle': data[1], 'eccentricity': data[2]}
        rh_visual = {'visual_area': data[3], 'polar_angle': data[4], 'eccentricity': data[5]}
        # Load the FreeSurfer subject:
        sub = ny.freesurfer_subject(os.path.join(freesurfer_path, subject))
        print(f'STEP: Retinotopic variables loaded for {subject}')
        return sub, lh_visual, rh_visual
    except:
        print(f'STEP: No precomputed variables found for {subject}. Generating new retinotopic map')
        # Load the FreeSurfer subject:
        sub = ny.freesurfer_subject(os.path.join(freesurfer_path, subject))
        
        # calculate benson 14 varea eccen and angle (this is equivalent to run the benson atlas and then importing the files e.g. lh.benson14_varea.mgz)
        (lh_visual, rh_visual) = ny.vision.predict_retinotopy(sub = sub, template = 'benson14', registration = 'fsaverage', sym_angle = False, names = 'properties')
   
        # Save session variables as pickle
        with open(session, 'wb') as f:
            pickle.dump([lh_visual, rh_visual], f)
        print(f'STEP:Retinotopic map saved for {subject}.')
        
        return sub, lh_visual, rh_visual
    
def make_fov(fov_radius_list, out_root, template_image, sub, lh_visual, rh_visual):
    for fov_radius in fov_radius_list:
        ## Define and merge (lh, rh) FOVEAL ROI
        roi_fov_lh = (lh_visual['visual_area'] == 1) & (lh_visual['eccentricity'] <= fov_radius)
        roi_fov_rh = (rh_visual['visual_area'] == 1) & (rh_visual['eccentricity'] <= fov_radius)
        roi_fov = sub.cortex_to_image((roi_fov_lh, roi_fov_rh), template_image)
        ## Make output dir
        out_path_fov = os.path.join(out_root, 'fov', str(fov_radius))
        try:
            os.makedirs(out_path_fov)
        except:
            pass
        ## Save ROIs
        fov_name = os.path.join(out_path_fov, (str(subject) + '_FOV_original.nii.gz'))
        ny.save(fov_name, roi_fov)
        print(f'DONE: {subject} {fov_name} saved')
    return


def make_per(per_radius_list, out_root, template_image, sub, lh_visual, rh_visual):    
    ## Define and merge PERIPHERAL and OPPOSITE ROI
    #  convert the polar angle from degrees centered at the UVM to standard polar coordinates
    (lh_geo_x, lh_geo_y) = ny.as_retinotopy(lh_visual, 'cartesian')
    (rh_geo_x, rh_geo_y) = ny.as_retinotopy(rh_visual, 'cartesian')
    # define visual field position (cartesian) from eccentricity and angle (polar angle)
    (stim1_normal_x, stim1_normal_y) = (-4.98, -4.98)
    (stim2_normal_x, stim2_normal_y) = (+4.98, +4.98)
    (stim1_inverted_x, stim1_inverted_y) = (-4.98, +4.98)
    (stim2_inverted_x, stim2_inverted_y) = (+4.98, -4.98)
    
    for per_radius in per_radius_list:
        # make ROIs based on position and radius size (per_radius)    
        roi_per_normal_rh = (rh_visual['visual_area'] == 1) & (np.sqrt((rh_geo_x - stim1_normal_x)**2 + (rh_geo_y - stim1_normal_y)**2) <= per_radius)
        roi_per_normal_lh = (lh_visual['visual_area'] == 1) & (np.sqrt((lh_geo_x - stim2_normal_x)**2 + (lh_geo_y - stim2_normal_y)**2) <= per_radius)
        roi_per_inverted_rh = (rh_visual['visual_area'] == 1) & (np.sqrt((rh_geo_x - stim1_inverted_x)**2 + (rh_geo_y - stim1_inverted_y)**2) <= per_radius)
        roi_per_inverted_lh = (lh_visual['visual_area'] == 1) & (np.sqrt((lh_geo_x - stim2_inverted_x)**2 + (lh_geo_y - stim2_inverted_y)**2) <= per_radius)
         
        # use surface_to_image to merge lh and rh
        roi_per_normal = sub.cortex_to_image((roi_per_normal_lh, roi_per_normal_rh), template_image)
        roi_per_inverted = sub.cortex_to_image((roi_per_inverted_lh, roi_per_inverted_rh), template_image)
        ## Make output dir
        out_path_per = os.path.join(out_root, 'per', str(per_radius))
        out_path_inv = os.path.join(out_root, 'inv', str(per_radius))
        try:
            os.makedirs(out_path_per)
        except:
            pass
        try:
            os.makedirs(out_path_inv)
        except:
            pass
        ## Save ROIs
        per_name = os.path.join(out_path_per, (str(subject) + '_PER_original.nii.gz'))
        opp_name = os.path.join(out_path_inv, (str(subject) + '_INV_original.nii.gz'))
        ny.save(opp_name, roi_per_normal)
        print(f'DONE: {subject} {per_name} saved')
        ny.save(per_name, roi_per_inverted)
        print(f'DONE: {subject} {opp_name} saved')
    return 

for subject in subjects:
    if subject.startswith('sub'):
        print(f'##### START: {subject} #####')
        session = os.path.join(root, 'derivatives', 'masks','pickled', (subject + '.pkl'))
        sub, lh_visual, rh_visual = get_retinotopic(subject, session)
        ## Import subject's raw image from FreeSurfer as template
        template_names = sorted(sub.images)
        template_image = sub.images['brain'] # raw: T1, brain: freesurfer
        template_image = ny.image_clear(template_image)
        ## Make ROIs
        print(f'STEP: Generating ROIs for {subject}.')
        make_fov(fov_radius_list, out_root, template_image, sub, lh_visual, rh_visual)
        make_per(per_radius_list, out_root, template_image, sub, lh_visual, rh_visual)
        
        


            
            




