"""
    Download the biased sessions of all brain-wide-map mice.
    This code requires an IBL environment, see: https://docs.internationalbrainlab.org/notebooks_external/data_download.html
"""
import pandas as pd
from one.api import ONE
import numpy as np
import json

bwm = json.load(open("bwm_names.json", 'r'))
data_folder = 'session_data/'

one = ONE()

for subject in bwm:
    try:
        trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

        # Load training status and join to trials table
        training = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')
        trials = (trials
                  .set_index('session')
                  .join(training.set_index('session'))
                  .sort_values(by='session_start_time', kind='stable'))
    except:
        print("Not working {}".format(subject))
        continue

    trials.to_parquet(data_folder + 'trials_{}.pqt'.format(subject))