def BIDS_events_to_SPM(sub_id, events_runs, out_root)
    """Summary line.

        This function wants a list of BIDS events DataFrames (columns:
        'trial_type', 'onset', 'duration') and saves a .mat file that
        can be used as a multicondition file in a SPM GLM scipt. The
        DataFrames must be ordered by run and belonging to the same subject
        e.g., [sub-01_run1, sub-01_run2, etc.]

        Args:
            sub_id (int): Subject id (e.g., '1' for sub-01, '2' for sub-02, etc.)
            events_runs (list): list of BIDS events DataFrames 
            out_root (str): root output folder

        Returns:
            Saves SPM multicondition file (*.mat) in ~/out_root/sub-{sub_id}

        """
    import itertools	import os
	import pandas as pd
	import numpy as np
	from scipy.io import savemat
	from util_func import get_timings
    
    sub = 'sub-' + str(sub_id).zfill(2)

    for i, events_run in enumerate(events_runs):
        df = events_run.rename(
            columns={'trial_type': 'names', 'onset': 'onsets', 'duration': 'durations'})
        new_df = pd.DataFrame(index=list(
            range(len(df['names'].unique()))), columns=df.columns)
        for j, name in enumerate(df['names'].unique()):
            filtered = df.loc[df['names'] == name]
            new_df['names'][j] = name
            new_df['onsets'][j] = list(filtered['onsets'].values)
            new_df['durations'][j] = list(filtered['durations'].values)
        mat_dict = {'names': np.array(new_df['names'], dtype=object), 'onsets': np.array(
            new_df['onsets'], dtype=object), 'durations': np.array(new_df['durations'], dtype=object)}

        filename = f'{sub}_run-{str(i+1)}.mat'
        sub_dir = os.path.join(out_root, sub)
        os.makedirs(sub_dir, exist_ok=True)
        file_out = os.path.join(sub_dir, filename)
        savemat(file_out, mat_dict)
        return
