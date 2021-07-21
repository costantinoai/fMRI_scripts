%-----------------------------------------------------------------------
% Job saved on 14-Jan-2021 21:09:58 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
clear
events_root = fullfile('E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\events\stimulus');
spm_root = fullfile('E:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\SPM');
% for classes = 1:2
classes = 2;
    for i=24:25
        clearvars -except classes i events_root spm_root
        sub = strcat("sub-",num2str(i, '%02d'));
        fprintf(strcat('########## \n STEP: running\t', sub,'\n########## \n'))

        sub_dir = fullfile(spm_root,sub);
        
        if classes == 1
            out_dir = fullfile(spm_root, 'RSA_blocks_1_lev_6HMP_all',sub);
        else
            out_dir = fullfile(spm_root, 'RSA_blocks_1_lev_6HMP_fv',sub);
        end

        runs = length(dir(fullfile(sub_dir, '*exp*.tsv')));

        spm('defaults','fmri');
        spm_jobman('initcfg');

        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(out_dir);
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        for run=1:runs
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).scans = spm_select('expand', fullfile(sub_dir,strcat(sub,'_task-exp_run-', string(run),'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii')));
            if classes == 1
                matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi = cellstr(fullfile(events_root, sub, strcat(sub, '_run-', string(run),'_BLOCKS.mat')));
            else
                matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi = cellstr(fullfile(events_root, sub, strcat(sub, '_run-', string(run),'_BLOCKS_fv.mat')));
            end
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress = struct('name', {}, 'val', {});
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi_reg = cellstr(fullfile(sub_dir, strcat(sub,'_task-exp_run-',string(run),'_desc-confounds_timeseries.txt')));
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).hpf = 128;
        end
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        spm_jobman('run',matlabbatch);
    end
% end
