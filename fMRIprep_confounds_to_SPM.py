def fMRIprep_confounds_to_SPM(sub_id, run_id, json_run, tsv_run, out_dir, pipeline):
    """Summary line.

    This function gets the json and tsv specific for the run and returns
    the confounds pandas dataframe.

    Args:
        sub_id (int): Subject id (e.g., '1' for sub-01, '2' for sub-02, etc.)
        run_id (int): Run number id
        json_run (dict): JSON dictionary of the selected run
        tsv_run (DataFrame): TSV pandas DataFrame of the selected run 
        out_dir (str): directory to save the DataFrame. The df is saved as a SPM-ready txt file
        pipeline (str): a string describing the pipeline. The string must 
            include denoising (form: str(int)-str) strategies separated by '_'.
            e.g., 'HMP-6_GS-4_SpikeReg_cosine'
            Possible strategies:
                HMP - Head motion parameters (6,12,24)
                GS - Global signal (1,2,4)
                Phys - Physiological noise (2,4,8)
                aCompCor - aCompCor (10,50)
                SpikeReg - motion outliers FD > 0.5, DVARS > 1.5
                cosine - Discrete cosine-basis regressors low frequencies -> HPF
                Null - returns a blank df

    Returns:
        DataFrame: pandas.DataFrame of selected regressors

    """
    import itertools
    import pandas as pd
    
    sub = 'sub-' + str(sub_id).zfill(2)
    
    pipeline_splits = pipeline.split('_')
    selected_keys = []
    for pipeline_split in pipeline_splits:
        try:
            conf_num, conf_split = pipeline_split.split('-')
        except:
            conf_split = pipeline_split
        if conf_split == 'Null':
            selected_keys = []
        if conf_split == 'HMP':  # Head Motion Parameters(6, 12 or 24)
            if int(conf_num) % 6 != 0:
                raise ValueError(
                    'Head Motion Parameters must be a multiple of 6 (rot_x,y,z; trans_x,y,z)')
            hmp_id = int(conf_num) // 6
            if hmp_id > 0:
                selected_keys.append(
                    ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])
            if hmp_id > 1:
                selected_keys.append(['rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                                      'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1'])
            if hmp_id > 2:
                selected_keys.append(['rot_x_power2', 'rot_y_power2', 'rot_z_power2', 'trans_x_power2', 'trans_y_power2', 'trans_z_power2', 'rot_x_derivative1_power2',
                                      'rot_y_derivative1_power2', 'rot_z_derivative1_power2', 'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2'])
        if conf_split == 'GS':  # Global Signal (raw, derivative, power)
            if int(conf_num) > 4:
                raise ValueError('Global signal must be <= 4')
            gs_id = int(conf_num)
            if gs_id > 0:
                selected_keys.append(['global_signal'])
            if gs_id > 1:
                selected_keys.append(['global_signal_derivative1'])
            if gs_id > 2:
                selected_keys.append(
                    ['global_signal_derivative1_power2, global_signal_power2'])
        # Physiological Noise (raw, derivative, power)
        if conf_split == 'Phys':
            if int(conf_num) % 2 != 0:
                raise ValueError(
                    'Phisiological regressors must be multiple of 2 (WM, CSF)')
            phys_id = int(conf_num) // 2
            if phys_id > 0:
                selected_keys.append(['white_matter', 'csf'])
            if phys_id > 1:
                selected_keys.append(
                    ['white_matter_derivative1', 'csf_derivative1'])
            if phys_id > 2:
                selected_keys.append(['white_matter_derivative1_power2',
                                      'csf_derivative1_power2', 'white_matter_power2', 'csf_power2'])
        # aCompCor (10: first 5 for each mask, 50: 50% of variance for each mask)
        if conf_split == 'aCompCor':
            csf_50_dict = {key: value for key, value in json_run.items() if (('Mask' in value) and (
                value['Mask'] == 'CSF') and (value['Method'] == 'aCompCor') and ('dropped' not in key))}
            wm_50_dict = {key: value for key, value in json_run.items() if (('Mask' in value) and (
                value['Mask'] == 'WM') and (value['Method'] == 'aCompCor') and ('dropped' not in key))}
            if (int(conf_num) != 10) and (int(conf_num) != 50):
                raise ValueError(
                    'aCompCorr can only be 10 (5 highest for each mask) or 50 (50% variance for each mask)')
            if int(conf_num) == 10:
                csf_10 = sorted(list(csf_50_dict.keys()))[0:5]
                wm_10 = sorted(list(wm_50_dict.keys()))[0:5]
                selected_keys.append(wm_10)
                selected_keys.append(csf_10)
            elif int(conf_num) == 50:
                csf_50 = list(csf_50_dict.keys())
                wm_50 = list(wm_50_dict.keys())
                selected_keys.append(wm_50)
                selected_keys.append(csf_50)
        if conf_split == 'cosine':
            cosine_keys = [key for key in tsv_run.columns if 'cosine' in key]
            selected_keys.append(cosine_keys)
        if conf_split == 'SpikeReg':
            motion_outlier_keys = [
                key for key in tsv_run.columns if 'motion_outlier' in key]
            selected_keys.append(motion_outlier_keys)
    selected_keys = list(itertools.chain.from_iterable(selected_keys))
    confounds_df = tsv_run[tsv_run.columns.intersection(selected_keys)].copy()
    
    filename_out = os.path.join(out_dir, f'{sub}_run-{str(run_id)}_desc-SPM-nuisance-regressors_pipeline-{pipeline}.txt')
    confounds_df.to_csv(filename_out, sep='\t', index=False, header=False)
    return confounds_df
