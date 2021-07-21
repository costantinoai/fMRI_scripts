# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:17:39 2021

@author: 45027900
"""

import os
import math
import glob
import numpy as np
import nibabel as nb
from scipy.io import loadmat
from seaborn import heatmap as hm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, sem, t
import pickle
from skimage.transform import resize

log = True
# wait_hours = 4
# ctime = time.ctime()
# print(f'START: {ctime}, sleeping for {str(wait_hours)} hours...')
# for half in range(wait_hours*2):
#     print(f'\t...{((wait_hours*2)-half)*30} minutes left')
#     time.sleep(60*30)
# print('...done!')


def digitize_rdm(rdm_raw, n_bins=10):
    """Digitize an input matrix to n bins (10 bins by default)
    rdm_raw: a square matrix 
    """
    # compute the bins
    rdm_bins = [np.percentile(np.ravel(rdm_raw), 100/n_bins * i)
                for i in range(n_bins)]
    # compute the vectorized digitized value
    rdm_vec_digitized = np.digitize(
        np.ravel(rdm_raw), bins=rdm_bins) * (100 // n_bins)
    # reshape to matrix
    rdm_digitized = np.reshape(rdm_vec_digitized, np.shape(rdm_raw))
    return rdm_digitized


def plot_RSMS(data_array, labels_df_list, title_list, sub):
    rows = 2
    columns = 4

    main_fig, axs = plt.subplots(rows, columns, figsize=(20, 10))
    main_fig.suptitle(f'{sub}\nRepresentational Dissimilarity Matrix')
    selector = int(str(rows) + str(columns) + '1')

    for i in range(data_array.shape[0]):
        data = digitize_rdm(data_array[i][0])
        plt.subplot(selector + i)
        # Plot
        ax = hm(data, vmin=0, vmax=100, cmap='jet', square=True)

        # Pull out the bin edges between the different categories
        line_width = 2
        bins = np.unique(np.array([label.split('-')[0]
                                   for label in labels_df_list[i].values]))
        binsize = np.histogram(
            list(range(len(labels_df_list[i]))), len(bins))[0]
        edges = np.concatenate([np.asarray([0]), np.cumsum(binsize)])
        labels_loc = [edges[i] + (binsize[i] / 2) for i in range(len(binsize))]
        ax.set_xticks(labels_loc)
        ax.set_xticklabels(bins, rotation=30)
        ax.set_yticks(labels_loc)
        ax.set_yticklabels(bins)
        ax.vlines(edges, 0, len(
            labels_df_list[i]), lw=line_width, color='black')
        ax.hlines(edges, 0, len(
            labels_df_list[i]), lw=line_width, color='black')
        ax.set_title(title_list[i])
    return


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, se


# Set folders and initialise variables
sns.set()
sns.color_palette("Paired")

beta_root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\SPM\RSA_blocks_1_lev_6HMP'
mask_root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\masks\original\original_realignedMNI\fov'
out_root = r'E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\MVPA\RSA_fov_allsizes'

conditions_list = [('car*', 'bike*', 'female*', 'male*'),
                   ('face*', 'vehicle*')]
rois = [os.path.split(glob.glob(os.path.join(mask_root, '*'))[i])[-1]
        for i in range(len(glob.glob(os.path.join(mask_root, '*'))))]

stats = {roi: {
    'category': {'t': [], 'p': [], 'same': {'avg': [], 'avg-h': [], 'avg+h': [], 'sem': [], 'n': []}, 'diff': {'avg': [], 'avg-h': [], 'avg+h': [], 'sem': [], 'n': []}},
    'class':    {'t': [], 'p': [], 'same': {'avg': [], 'avg-h': [], 'avg+h': [], 'sem': [], 'n': []}, 'diff': {'avg': [], 'avg-h': [], 'avg+h': [], 'sem': [], 'n': []}}
} for roi in rois}
correlations = {sub_id: {level: {roi: pd.DataFrame(columns=(
    'corr', 'cat1', 'cat2')) for roi in rois} for level in stats[rois[0]].keys()} for sub_id in range(2, 26)}

for conditions in conditions_list:
    rois_dirs = glob.glob(os.path.join(mask_root, '?.?'))
    # point to the right folder
    if ('face*' in conditions) or ('vehicle*' in conditions):
        beta_subdir = beta_root + '_fv'
        level = 'class'
    else:
        beta_subdir = beta_root + '_all'
        level = 'category'

    rdm_avg_all = np.zeros(
        (len(rois), 24, len(conditions) * 5, len(conditions) * 5))
    out_dir = os.path.join(out_root, level)
    if log:
        os.makedirs(out_dir, exist_ok=True)
    for sub_id in range(2, 26):
        sub = 'sub-' + str(sub_id).zfill(2)
        beta_loc = os.path.join(beta_subdir, sub)
        print(f'#### START: {sub} ####')

        print('STEP: loading masks...', end='\r')
        # define masks locations
        mask_files = {roi: glob.glob(os.path.join(
            mask_root, roi, 'Linear', 'resampled_nearest', sub + '*FOV*'))[0] for roi in rois}
        print('done!')

        # generate betas mapping dataframe
        print('STEP: generating beta mapping dataframe...', end='\r')
        betas_df = pd.DataFrame(None, columns=(
            'beta_path', 'spm_filename', 'spm_class', 'spm_run', 'bin', 'array'))
        betas_df['beta_path'] = sorted(
            glob.glob(os.path.join(beta_loc, '*beta*.?ii')))

        mat = loadmat(os.path.join(beta_loc, 'SPM.mat'))
        beta_names = [str(mat['SPM']['Vbeta'][0][0][0][beta_id][0][0]) for beta_id in range(len(betas_df['beta_path'])) if str(
            mat['SPM']['Vbeta'][0][0][0][beta_id][0][0]) == os.path.basename(betas_df['beta_path'][beta_id])]
        beta_classes = [str(mat['SPM']['Vbeta'][0][0][0][beta_id][5][0]).split(' ')[-1].split('*')[0] for beta_id in range(len(
            betas_df['beta_path'])) if str(mat['SPM']['Vbeta'][0][0][0][beta_id][0][0]) == os.path.basename(betas_df['beta_path'][beta_id])]
        beta_runs = [str(mat['SPM']['Vbeta'][0][0][0][beta_id][5][0]).split(' ')[-2].split('(')[-1][0] for beta_id in range(len(
            betas_df['beta_path'])) if str(mat['SPM']['Vbeta'][0][0][0][beta_id][0][0]) == os.path.basename(betas_df['beta_path'][beta_id])]
        beta_bins = [str(mat['SPM']['Vbeta'][0][0][0][beta_id][5][0]).split(' ')[-1].split('(')[-1][0] for beta_id in range(len(betas_df['beta_path']))
                     if str(mat['SPM']['Vbeta'][0][0][0][beta_id][0][0]) == os.path.basename(betas_df['beta_path'][beta_id])]

        betas_df['spm_filename'] = beta_names
        betas_df['spm_class'] = beta_classes
        betas_df['spm_run'] = beta_runs
        betas_df['bin'] = beta_bins

        betas_df = betas_df[betas_df.spm_class.str.match(
            "|".join(list(conditions)))]
        print('done!')

        beta_img = [nb.load(img).get_fdata() for img in betas_df['beta_path']]
        betas_df['array'] = beta_img

        betas_df = betas_df.reset_index(drop=True)
        runs = np.unique(betas_df['spm_run'].values)

        # remove the mean value in all conditions for each voxel in a run (cocktail-blank removal)
        print('STEP: removing average pattern (mean value in the run for each voxel)...', end='\r')
        all_betas_data = np.array([array for array in betas_df['array']])
        for run in runs:
            betas_idx = betas_df['spm_run'] == run
            run_array = all_betas_data[betas_idx.values]
            run_array_avg = np.array([np.mean(run_array, 0)
                                      for rows in range(run_array.shape[0])])
            run_array_demeaned = run_array - run_array_avg

            list_id = 0
            for index, beta_idx in enumerate(betas_idx):
                if beta_idx:
                    betas_df.iloc[index]['array'] = run_array_demeaned[list_id]
                    list_id += 1

        print('done!')

        # select and plot data
        rdm_sub = np.zeros(
            (len(mask_files.keys()), 1, len(conditions) * len(runs), len(conditions) * len(runs)))
        labels_roi = []
        titles_roi = []
        betas_df1 = betas_df.sort_values(
            ['spm_class', 'spm_run'], axis=0).reset_index(drop=True)

        for roi_idx, roi in enumerate(mask_files.keys()):
            # import mask and mask data
            mask = nb.load(mask_files[roi]).get_fdata() > 0
            betas_masked = [df_array[mask]
                            for df_array in betas_df1['array']]

            # compute correlation matrix (nans are masked by pandas correlation function)
            print(
                f'STEP: calculating correlation matrix for {sub}, {roi}...')
            beta = pd.DataFrame(None)

            matrix_diagonal = 0
            cell_idx = 0
            for i in range(len(betas_masked)):
                for j in range(len(betas_masked)):
                    if (len(betas_masked)*i+j) % 50 == 0:
                        print(
                            f'\t... {len(betas_masked)*i + j}/{len(betas_masked)*len(betas_masked)} - {betas_df1["spm_filename"][i]}, {betas_df1["spm_filename"][j]}\t({betas_df1["spm_class"][i]}, {betas_df1["spm_class"][j]})')
                    beta[0] = betas_masked[i]
                    beta[1] = betas_masked[j]
                    # correlation
                    corr = beta.corr()
                    rdm_sub[roi_idx, 0, i, j] = 1 - corr[0][1]
                    # # euclidean
                    # corr = euclidean(beta[0][~np.isnan(beta[0])],
                    #                  beta[1][~np.isnan(beta[1])])
                    # rdm_sub[roi_idx, 0, i, j] = corr
                    # store results
                    if i == j:
                        matrix_diagonal = j + 1
                    if (betas_df1['spm_run'][i] != betas_df1['spm_run'][j]) and (j < matrix_diagonal):
                        tmp_df = pd.DataFrame(
                            index=[0], columns=('idx', 'corr', 'cat1', 'cat2'))
                        tmp_df['idx'][0] = cell_idx
                        cell_idx += 1
                        tmp_df['corr'][0] = corr[0][1]
                        tmp_df['cat1'][0] = betas_df1['spm_class'][i]
                        tmp_df['cat2'][0] = betas_df1['spm_class'][j]
                        correlations[sub_id][level][roi] = correlations[sub_id][level][roi].append(
                            tmp_df)
            rdm_avg_all[roi_idx, sub_id - 2] = rdm_sub[roi_idx, 0]
            labels_roi.append(betas_df1['spm_class'])
            titles_roi.append(f'ROI: {roi} (percentile bins)')
            print('...done!')

        print(
            f'STEP: plotting and saving matrices for {sub} ...', end='\r')
        plot_RSMS(rdm_sub, labels_roi, titles_roi, sub)
        if log:
            filename = os.path.join(out_dir, f'{sub}_blocks.png')
            plt.savefig(filename)
        plt.show()
        print('...done!')

    # save averages array to file
    if log:
        array_filename = os.path.join(out_dir, 'averages_array.npy')
        np.save(array_filename, rdm_avg_all)

    # compute and plot grand average array
    rdm_avg_all_avg = np.expand_dims(np.nanmean(rdm_avg_all, axis=1), 1)
    labels_list_all = [betas_df1['spm_class']
                       for roi in range(rdm_avg_all_avg.shape[0])]
    title_list_all = list(mask_files.keys())

    plot_RSMS(rdm_avg_all_avg, labels_list_all, title_list_all,
              'All subjects')
    if log:
        filename = os.path.join(out_dir, 'grand-average_all-subjects.png')
        plt.savefig(filename)
    plt.show()

    # STATS
    # compute long format matrix with correlations
    corr_df = pd.DataFrame(None)
    for roi in rois:
        for sub_id in range(2, 26):
            tmp_df = pd.DataFrame(
                columns=('idx', 'corr', 'class_id', 'roi', 'sub_id', 'category'))
            tmp_df['idx'] = correlations[sub_id][level][roi]['idx'].values
            tmp_df['corr'] = correlations[sub_id][level][roi]['corr'].values
            tmp_df['class_id'] = correlations[sub_id][level][roi]['cat1'].values + \
                '_' + correlations[sub_id][level][roi]['cat2'].values
            tmp_df['roi'] = roi
            tmp_df['sub_id'] = str(sub_id).zfill(2)
            tmp_df['category'] = correlations[sub_id][level][roi]['cat1'].values == correlations[sub_id][level][roi]['cat2'].values
            tmp_df = tmp_df.reset_index(drop=True)
            corr_df = corr_df.append(tmp_df).reset_index(drop=True)
    if log:
        filename = os.path.join(out_dir, 'corr_df.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(corr_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # do stats (t-test or ANOVA?) on correlations - this tells us if same (bike bike) is significantly diff from diff (bike, car)
    # collect stats
    for roi_idx, roi in enumerate(rois):
        diff = corr_df.loc[(corr_df['category'] == False)
                           & (corr_df['roi'] == roi)]['corr'].values
        same = corr_df.loc[(corr_df['category'] == True)
                           & (corr_df['roi'] == roi)]['corr'].values
        stats[roi][level]['same']['avg'], stats[roi][level]['same']['avg-h'], stats[roi][level][
            'same']['avg+h'], stats[roi][level]['same']['sem'] = mean_confidence_interval(same)
        stats[roi][level]['same']['n'] = same.shape[0]
        stats[roi][level]['diff']['avg'], stats[roi][level]['diff']['avg-h'], stats[roi][level]['diff'][
            'avg+h'], stats[roi][level]['diff']['sem'] = mean_confidence_interval(diff)
        stats[roi][level]['diff']['n'] = diff.shape[0]
        stats[roi][level]['t'], stats[roi][level]['p'] = ttest_ind(
            same, diff, alternative='greater')
    # save stats dictionary
    if log:
        filename = os.path.join(out_root, 'stats-dict.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # make plot
    same_avg = [stats[roi][level]['same']['avg'] for roi in stats.keys()]
    diff_avg = [stats[roi][level]['diff']['avg'] for roi in stats.keys()]
    same_yerr = [stats[roi][level]['same']['sem'] for roi in stats.keys()]
    diff_yerr = [stats[roi][level]['diff']['sem'] for roi in stats.keys()]
    ps = [stats[roi][level]['p'] for roi in stats.keys()]
    hue = ['same'] * len(same_avg) + ['diff'] * len(diff_avg)
    x = list(stats.keys()) + (list(stats.keys()))
    y = same_avg + diff_avg
    sns.set(font_scale=1.4)
    fig_dims = (15, 9)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(x, y, hue, ax=ax, palette="Paired")
    # make error bars
    x_pos = np.array([[ax.get_xticks()[i] - 0.2, + ax.get_xticks()[i] + 0.2]
                      for i in range(len(ax.get_xticks()))]).flatten()
    y_pos = np.array([[same_avg[i], diff_avg[i]]
                      for i in range(len(same_avg))]).flatten()
    yerr = np.array([[same_yerr[i], diff_yerr[i]]
                     for i in range(len(diff_yerr))]).flatten()
    plt.errorbar(x=x_pos, y=y_pos, yerr=yerr,
                 fmt='none', c='black', capsize=5)
    ax.set_xticklabels(list(stats.keys()), rotation=25,
                       horizontalalignment='right', fontweight='light')
    ax.set_ylabel("Similarity (Pearson's r)")
    ax.set_xlabel('ROI')
    ax.set_title(f"Average Correlation (Pearson's r)\nLevel: {level}\n")

    for i, p in enumerate(ps):
        if p < 0.001:
            displaystring = r'***'
        elif p < 0.01:
            displaystring = r'**'
        elif p < 0.05:
            displaystring = r'*'
        else:
            displaystring = r''
        height = np.max([same_avg[i] + same_yerr[i],
                         diff_avg[i] + diff_yerr[i]]) + 0.02
        plt.text(list(range(len(same_avg)))[
                 i], height, displaystring, ha='center', va='center', size=20)
    # Show graphic
    plt.legend()
    plt.tight_layout()
    if log:
        filename = os.path.join(out_dir, f'corr_ttest_{level}.png')
        plt.savefig(filename)
    plt.show()
