"""
    Process the downloaded sessions in whichever way necessary - mostly reducing them to the current sessions of interest.
    Also computing session numbers.
"""
import pandas as pd
import json
import numpy as np

bwm = json.load(open("bwm_names.json", 'r'))
data_folder = 'session_data/'
processed_folder = 'processed_data/'

num_of_used_sessions = []  # keep track of how many sessions each mouse has under the current processing
all_dfs = []  # also make one big dataframe to save as csv
all_plus_trained_dfs = []
num_of_ready_sessions = []
num_of_delay_sessions = []

def augment_df(sessions, subject):
    dates = np.unique(sessions['session_start_time'])
    date_map = dict(zip(dates, range(len(dates))))
    sessions = sessions.assign(session_number=sessions['session_start_time'].map(date_map))  # add a column of session numbers (for the selected subset of sessions)

    sessions.to_parquet(processed_folder + 'processed_trials_{}.pqt'.format(subject))

    sessions = sessions.assign(subject=subject)  # add name for big df

    return sessions

for subject in bwm:
    df = pd.read_parquet(data_folder + 'trials_{}.pqt'.format(subject))

    ephys_sessions = df[df['task_protocol'].str.startswith('_iblrig_tasks_ephys')]  # subselect to ephys sessions
    ephys_plus_trained_sessions = df[np.logical_or(df.training_status == 'ready4ephysrig',  # subselect a few more sessions
                                     np.logical_or(df.training_status == 'ready4delay',
                                                   df['task_protocol'].str.startswith('_iblrig_tasks_ephys')))]

    all_dfs.append(augment_df(ephys_sessions, subject))
    all_plus_trained_dfs.append(augment_df(ephys_plus_trained_sessions, subject))

    num_of_used_sessions.append(len(np.unique(ephys_sessions.session_start_time)))
    num_of_ready_sessions.append(len(np.unique(ephys_plus_trained_sessions.session_start_time)))

uber_df = pd.concat(all_dfs)
uber_df.to_csv(processed_folder + "all_mice.csv")

uber_plus_trained_df = pd.concat(all_plus_trained_dfs)
uber_plus_trained_df.to_csv(processed_folder + "all_mice_plus_trained.csv")

print("Number of sessions:")
print(np.unique(num_of_used_sessions, return_counts=1))
# Number of sessions:
# (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 16]), array([ 2, 13, 26, 24, 32, 15,  9,  5,  6,  2,  1,  2,  1,  1]))
print(sum(num_of_used_sessions))

print("Number of extended sessions")
print(np.unique(num_of_ready_sessions, return_counts=1))
# (array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 18]), array([ 1,  2, 16, 25, 28, 27, 14,  8,  5,  6,  2,  1,  2,  1,  1]))
print(sum(num_of_ready_sessions))