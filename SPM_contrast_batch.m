clear

for i=2:25
    clearvars -except i
    subject = strcat("sub-",num2str(i, '%02d'));
    fprintf(strcat('########## \n STEP: running \t', subject,'\n########## \n'))
    
    spm('defaults','fmri');
    spm_jobman('initcfg');

    matlabbatch{1}.spm.stats.con.spmmat = cellstr(strcat('D:\Andrea\fov\BIDS\derivatives\SPM_loc\1_lev_mr_smooth\', subject,'\SPM.mat'));
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'Faces > Vehicles';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1 -1 0];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'Vehicles > Faces';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = [-1 1 0];
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'repl';
    matlabbatch{1}.spm.stats.con.delete = 1;

    spm_jobman('run',matlabbatch);

end

