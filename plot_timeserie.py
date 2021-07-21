# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:35:55 2021

@author: 45027900
"""

import os, pickle, glob, itertools, json
from itertools import compress
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import signal
from nilearn import surface
import neuropythy as ny
from util_func import get_timings, get_confounds
# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

sns.set_style('darkgrid')

def plot_voxel(roi_list, voxel_n, events, selected_run = None):
    fig_data = pd.DataFrame(None, columns = ('timepoint','signal', 'run'))
    fig_data['signal'] = list(itertools.chain.from_iterable([run.T[voxel_n] for run in roi_list]))
    run_idx = []
    timepoint_idx = []
    for i, run in enumerate(roi_list):
            for j, tp in enumerate(run.T[voxel_n]):
                  run_idx.append(i+1)
                  timepoint_idx.append(j * 2)
    # ymin = min([np.amin(run) for run in roi_list])
    # ymax = max([np.amax(run) for run in roi_list])
    fig_data['run'] = run_idx
    fig_data['timepoint'] = timepoint_idx
    sns.set_style('darkgrid')
    if selected_run is not None:
        fig_data = fig_data[fig_data['run'] == selected_run]
    fig = sns.lineplot(data = fig_data, x = 'timepoint', y = 'signal', hue = 'run', palette = 'Set2')
    fig.xaxis.set_ticks(events[0]['onset'].values.astype(int) + 8)
    if selected_run is not None:
        fig.xaxis.set_ticklabels(events[selected_run - 1]['trial_type'].values.astype(str), rotation = 30)
    else:
        fig.set_xticklabels(labels = events[0]['onset'].values.astype(int) + 8, rotation = 30)
    # fig.set(ylim=(ymin, ymax))
    return fig

def plot_roi_avg(roi_list, events, selected_run = None):
    fig_data = pd.DataFrame(None, columns = ('timepoint','signal', 'run'))
    fig_data['signal'] = list(itertools.chain.from_iterable([np.average(run, axis = 1) for run in roi_list]))
    run_idx = []
    timepoint_idx = []
    for i, run in enumerate(roi_list):
            for j, tp in enumerate(run):
                  run_idx.append(i+1)
                  timepoint_idx.append(j * 2)
    # ymin = min([np.amin(run) for run in roi_list])
    # ymax = max([np.amax(run) for run in roi_list])
    fig_data['run'] = run_idx
    fig_data['timepoint'] = timepoint_idx
    sns.set_style('darkgrid')
    if selected_run is not None:
        fig_data = fig_data[fig_data['run'] == selected_run]
    fig = sns.lineplot(data = fig_data, x = 'timepoint', y = 'signal', hue = 'run', palette = 'Set2')
    fig.xaxis.set_ticks(events[0]['onset'].values.astype(int) + 8)
    if selected_run is not None:
        fig.xaxis.set_ticklabels(events[selected_run - 1]['trial_type'].values.astype(str), rotation = 30)
    else:
        fig.set_xticklabels(labels = events[0]['onset'].values.astype(int) + 8, rotation = 30)

    # fig.set(ylim=(ymin, ymax))
    return fig

def plot_roi_avg_tuple(roi_lists, events, selected_run = None):
    fig_data = pd.DataFrame(None, columns = ('timepoint','signal', 'run', 'denoising'))
    signal_run = []
    run_idx = []
    line_idx = []
    timepoint_idx = []
    for ii, pl_data in enumerate(roi_lists):
        signal_run.append(list(itertools.chain.from_iterable([np.average(run, axis = 1) for run in pl_data])))
        for i, run in enumerate(pl_data):
                for j, tp in enumerate(run):
                      run_idx.append(i+1)
                      timepoint_idx.append(j * 2)
                      line_idx.append(ii)
    ymin = -2 # min([np.amin(run) for run in pl_data])
    ymax = 2 # max([np.amax(run) for run in pl_data])
    fig_data['signal'] = list(itertools.chain.from_iterable(signal_run))
    fig_data['run'] = run_idx
    fig_data['timepoint'] = timepoint_idx
    fig_data['denoising'] = line_idx 
    sns.set_style('darkgrid')
    if selected_run is not None:
        fig_data = fig_data[fig_data['run'] == selected_run]
    fig = sns.lineplot(data = fig_data, x = 'timepoint', y = 'signal', hue = 'denoising')
    fig.xaxis.set_ticks(events[0]['onset'].values.astype(int) + 8)
    if selected_run is not None:
        fig.xaxis.set_ticklabels(events[selected_run - 1]['trial_type'].values.astype(str), rotation = 30)
    else:
        fig.set_xticklabels(labels = events[0]['onset'].values.astype(int) + 8, rotation = 30)
    fig.set(ylim=(ymin, ymax))
    # set alpha for marker (index 0) and for the rest line (indeces 3-6) 
    fig.get_children()[1]._alpha = 0.6
    fig.get_children()[1]._color = (125/225,25/225,0)
    fig.get_children()[0]._alpha = 0.7
    fig.get_children()[0]._color = (0,125/225,125/225)
    return fig

## DEFINE FOLDERS AND VARIABLES
# define folders
bids_root = os.path.abspath(r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS')
fmriprep_root = os.path.join(bids_root, 'derivatives', 'fmriprep_fsaverage', 'fmriprep')
events_root = os.path.join(bids_root, 'derivatives', 'events', 'all')
masks_root = os.path.join(bids_root, 'derivatives', 'masks', 'pickled')

pipelines = ['6-HMP_SpikeReg_cosine', '6-HMP_8-Phys_SpikeReg_cosine','24-HMP_SpikeReg_cosine', '6-HMP_10-aCompCor_SpikeReg_cosine', '24-HMP_10-aCompCor_SpikeReg_cosine', '24-HMP_8-Phys_SpikeReg_cosine']

for sub_id in range(2,26):
    sub = 'sub-' + str(sub_id).zfill(2)
    conditions = ('bike','car','female','male', 'rest')
    
    ## IMPORT DATA
    # import retinotopic maps
    mask_file = os.path.join(masks_root, (sub + '.pkl'))
    with open(mask_file, 'rb') as f:  
        data = pickle.load(f)
    lh_visual = {'visual_area': data[0], 'polar_angle': data[1], 'eccentricity': data[2]}
    rh_visual = {'visual_area': data[3], 'polar_angle': data[4], 'eccentricity': data[5]}
    # import functional gifti data (even: lh, odds: rh)
    giftis_files = sorted(glob.glob(os.path.join(fmriprep_root, sub, 'func', '*exp*fsnative*.func.gii')))
    time_series = [surface.load_surf_data(gifti).T for gifti in giftis_files]
    # import events
    events = get_timings(sub, conditions, duration='block')
    orientation = events[0]['orientation'][0]
    # import nuisance regressors
    conf_tsv_files = sorted(glob.glob(os.path.join(fmriprep_root, sub, 'func', '*exp*confounds*tsv')))
    conf_tsvs = [pd.read_csv(conf_tsv_file, sep='\t').fillna(0) for conf_tsv_file in conf_tsv_files]
    
    conf_json_files = sorted(glob.glob(os.path.join(fmriprep_root, sub, 'func', '*exp*confounds*json')))
    conf_jsons = []
    for conf_json_file in conf_json_files:
        with open(conf_json_file) as json_file:
            conf_jsons.append(json.load(json_file))
            
    confounds = {key: [] for key in pipelines}
    for pipeline in pipelines:
        for run_id in range(len(conf_tsvs)):
            confounds[pipeline].append(get_confounds(conf_jsons[run_id], conf_tsvs[run_id], pipeline))

    ## GENERATE ROIS ON SUBJECT SURFACE
    fov_radius = 4
    per_radius = 1.5
    rh_visual['polar_angle'] = rh_visual['polar_angle'] * -1 # this is because polar angle in the rh (left visual field) is positive (lh and rh are simmetrical)
    # foveal
    roi_fov_lh = (lh_visual['visual_area'] == 1) & (lh_visual['eccentricity'] <= fov_radius)
    roi_fov_rh = (rh_visual['visual_area'] == 1) & (rh_visual['eccentricity'] <= fov_radius)
    # convert the polar angle from degrees (visual, 0 deg is UVM, positive on right vf -> lh) to geographical (x,y are in deg)
    (lh_geo_x, lh_geo_y) = ny.as_retinotopy(lh_visual, 'geographical')
    (rh_geo_x, rh_geo_y) = ny.as_retinotopy(rh_visual, 'geographical')
    # now rh_geo_x has negative values (points on the left visual field) and lh_geo_x positive values (right visual field)
    # define visual field position (cartesian) from eccentricity and angle (polar angle)
    (stim1_normal_x, stim1_normal_y) = (-4.98, -4.98)
    (stim2_normal_x, stim2_normal_y) = (+4.98, +4.98)
    (stim1_inverted_x, stim1_inverted_y) = (-4.98, +4.98)
    (stim2_inverted_x, stim2_inverted_y) = (+4.98, -4.98)
    # peripheral
    roi_per_normal_rh = (rh_visual['visual_area'] == 1) & ((rh_geo_x - stim1_normal_x)**2 + (rh_geo_y - stim1_normal_y)**2 <= per_radius**2)
    roi_per_normal_lh = (lh_visual['visual_area'] == 1) & ((lh_geo_x - stim2_normal_x)**2 + (lh_geo_y - stim2_normal_y)**2 <= per_radius**2)
    roi_per_inverted_rh = (rh_visual['visual_area'] == 1) & ((rh_geo_x - stim1_inverted_x)**2 + (rh_geo_y - stim1_inverted_y)**2 <= per_radius**2)
    roi_per_inverted_lh = (lh_visual['visual_area'] == 1) & ((lh_geo_x - stim2_inverted_x)**2 + (lh_geo_y - stim2_inverted_y)**2 <= per_radius**2)
    
    if 0 in (sum(roi_fov_rh), sum(roi_fov_lh), sum(roi_per_normal_rh), sum(roi_per_normal_lh), sum(roi_per_inverted_rh), sum(roi_per_inverted_lh)):
        raise ValueError('One or more ROI have 0 vertices.')
    else:
        roi_info = pd.DataFrame(columns=('LH', 'RH'), index=('FOV','NORM','INV'))
        roi_info['LH'] = [sum(roi_fov_lh), sum(roi_per_normal_lh), sum(roi_per_inverted_lh)]
        roi_info['RH'] = [sum(roi_fov_rh), sum(roi_per_normal_rh), sum(roi_per_inverted_rh)]
        print('#### ROI INFO ####\n',roi_info)
        
    ## GET VERTICES LISTS AND TIME SERIES IN ROIS
    # get vertices lists
    vertices_fov_lh = list(np.where(roi_fov_lh)[0])
    vertices_fov_rh = list(np.where(roi_fov_rh)[0])
    vertices_per_normal_lh = list(np.where(roi_per_normal_lh)[0])
    vertices_per_normal_rh = list(np.where(roi_per_normal_rh)[0])
    vertices_per_inverted_lh = list(np.where(roi_per_inverted_lh)[0])
    vertices_per_inverted_rh = list(np.where(roi_per_inverted_rh)[0])
    if (not set(vertices_fov_lh).isdisjoint(vertices_per_normal_lh)) | (not set(vertices_fov_lh).isdisjoint(vertices_per_inverted_lh)) | (not set(vertices_per_normal_lh).isdisjoint(vertices_per_inverted_lh)):
        raise ValueError('There is an overlap between foveal and peripheral vertices in the LEFT hemisphere')
    if (not set(vertices_fov_rh).isdisjoint(vertices_per_normal_rh)) | (not set(vertices_fov_rh).isdisjoint(vertices_per_inverted_rh)) | (not set(vertices_per_normal_rh).isdisjoint(vertices_per_inverted_rh)):
        raise ValueError('There is an overlap between foveal and peripheral vertices in the RIGHT hemisphere')
    print('There is no overlap between foveal, normal and peripheral ROIs\n')
    # get masked time series
    ts_fov_lh = []
    ts_fov_rh = []
    ts_per_normal_lh = []
    ts_per_normal_rh = []
    ts_per_inverted_lh = []
    ts_per_inverted_rh = []
    
    for i,time_serie in enumerate(time_series):
        if i % 2 == 0: # if the time_serie is left hemisphere
            ts_fov_lh.append(np.array([list(compress(time_serie[timepoint], roi_fov_lh)) for timepoint in range(len(time_serie))]))
            ts_per_normal_lh.append(np.array([list(compress(time_serie[timepoint], roi_per_normal_lh)) for timepoint in range(len(time_serie))]))
            ts_per_inverted_lh.append(np.array([list(compress(time_serie[timepoint], roi_per_inverted_lh)) for timepoint in range(len(time_serie))]))
        else:
            ts_fov_rh.append(np.array([list(compress(time_serie[timepoint], roi_fov_rh)) for timepoint in range(len(time_serie))]))
            ts_per_normal_rh.append(np.array([list(compress(time_serie[timepoint], roi_per_normal_rh)) for timepoint in range(len(time_serie))]))
            ts_per_inverted_rh.append(np.array([list(compress(time_serie[timepoint], roi_per_inverted_rh)) for timepoint in range(len(time_serie))]))
    
    ## MERGE RH AND LH horizontally and assign to stim/opp roi
    ts_fov = [np.concatenate((ts_fov_lh[run], ts_fov_rh[run]), axis = 1) for run in range(len(ts_fov_lh))]
    if orientation == 'inverted':
        ts_per_stim = [np.concatenate((ts_per_inverted_lh[run], ts_per_inverted_rh[run]), axis = 1) for run in range(len(ts_per_inverted_lh))]
        ts_per_opp = [np.concatenate((ts_per_normal_lh[run], ts_per_normal_rh[run]), axis = 1) for run in range(len(ts_per_normal_lh))]
    elif orientation == 'normal':
        ts_per_opp = [np.concatenate((ts_per_inverted_lh[run], ts_per_inverted_rh[run]), axis = 1) for run in range(len(ts_per_inverted_lh))]
        ts_per_inv = [np.concatenate((ts_per_normal_lh[run], ts_per_normal_rh[run]), axis = 1) for run in range(len(ts_per_normal_lh))]

    # ## CLEAN and plot DATA (DETRENDING, NORMALIZATION) 
    # # HPF: 0.007 (3 cycles every run); LPF: 0.1 (neuronal signal is below 6-8 cycle))
    scan_time = ts_fov[0].shape[0] * 2
    bold_peak = 6.6
    lpf = round(((scan_time/bold_peak) / scan_time), 2)
    
    # ts_fov_scaled = [signal.clean(run, t_r=2, detrend=True) for run in ts_fov]
    # ts_per_stim_scaled = [signal.clean(run, t_r=2, detrend=True) for run in ts_per_stim]
    # ts_per_opp_scaled = [signal.clean(run, t_r=2, detrend=True) for run in ts_per_opp]
    
    # TODO: explore rh and lf 
    cosine_conf = [get_confounds(conf_jsons[run_id], conf_tsvs[run_id], 'cosine') for run_id in range(len(conf_tsvs))]
    ## EXPLORE DATA
    for run_id in range(len(events)):
        plt_grid = plt.figure(figsize=(25,10))
        plt_grid.suptitle(f'{sub} - Run {str(run_id + 1)}', fontsize=16)
        plt.subplot(241)
        plt.title('Scaled + detrend')
        ts_per_stim_scaled = [signal.clean(run, t_r=2, detrend=True) for run in ts_per_stim]
        ts_per_stim_scaled_lpf = [signal.clean(run, t_r=2, detrend=True, low_pass = lpf) for run in ts_per_stim]
        plot_roi_avg_tuple((ts_per_stim_scaled,ts_per_stim_scaled_lpf), events, selected_run = run_id+1).set_xlabel('')
        plt.subplot(245)
        plt.title('Scaled + detrend + cosine')
        ts_per_stim_cos = [signal.clean(run, t_r=2, detrend=True, confounds = cosine_conf[run_id]) for run in ts_per_stim]
        ts_per_stim_cos_lpf = [signal.clean(run, t_r=2, detrend=True, confounds = cosine_conf[run_id], low_pass = lpf) for run in ts_per_stim]
        plot_roi_avg_tuple((ts_per_stim_scaled,ts_per_stim_cos_lpf), events, selected_run = run_id+1).set_xlabel('')
        sub_plt_id = 241
        for pipeline in pipelines:
            sub_plt_id += 1
            if sub_plt_id == 245:
                sub_plt_id += 1
            plt.subplot(sub_plt_id)
            plt.title(f'{pipeline}')
            ts_per_stim_denoised = [signal.clean(run, t_r=2, detrend=False, confounds = confounds[pipeline][run_id]) for run in ts_per_stim]
            ts_per_stim_denoised_lpf = [signal.clean(run, t_r=2, detrend=False, confounds = confounds[pipeline][run_id], low_pass = lpf) for run in ts_per_stim]
            plot_roi_avg_tuple((ts_per_stim_denoised,ts_per_stim_denoised_lpf), events, selected_run = run_id+1).set_xlabel('')
        plt.show()
   






