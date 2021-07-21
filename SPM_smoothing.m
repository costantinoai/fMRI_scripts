%% smoothing
clear
for i=2:25
    clearvars -except i
    index = i-1;
    fprintf(strcat('########## \n STEP: running sub-', string(i),'\n########## \n'))
    sub = sprintf('sub-%02i',i);
    spm('defaults','fmri');
    spm_jobman('initcfg');
    files = cellstr(spm_select('expand', fullfile(strcat('D:\Andrea\fov\BIDS\derivatives\SPM\',sub,'\'),strcat(sub,'_task-loc_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'))));
    matlabbatch{1}.spm.spatial.smooth.data = files;
    matlabbatch{1}.spm.spatial.smooth.fwhm = [5 5 5];
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    matlabbatch{1}.spm.spatial.smooth.im = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 'smooth_';
    
    spm_jobman('run',matlabbatch);
end

