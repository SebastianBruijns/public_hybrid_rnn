"""
	Get some info about the dataset we're using, like number of animals, sessions, and trials.
	Also number of unique sessions and how they are split across train-test split.
"""
import pandas as pd
import numpy as np
import json
import pickle
import load_data
import matplotlib.pyplot as plt
import seaborn as sns

file="./processed_data/all_mice.csv"
alt_set = False

big_types = [np.array([0.    , 0.25  , 0.    , 0.    , 0.    , 0.    , 0.0625, 1.    ,
       0.125 , 0.    , 0.0625, 0.125 , 0.0625, 0.0625, 0.125 , 0.0625,
       0.0625, 0.25  , 0.125 , 0.    , 0.0625, 0.25  , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.25  , 0.125 , 0.    ]),
		np.array([0.    , 1.    , 1.    , 0.    , 0.    , 1.    , 0.125 , 0.    ,
       0.0625, 1.    , 0.    , 0.25  , 0.    , 0.0625, 0.    , 0.    ,
       0.    , 0.    , 0.    , 1.    , 0.0625, 0.25  , 0.    , 0.    ,
       0.    , 0.125 , 0.25  , 0.    , 0.    , 0.    ]),
		np.array([1.    , 0.    , 0.25  , 0.    , 0.    , 0.    , 0.    , 0.    ,
       1.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.125 ,
       0.25  , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.0625, 0.25  , 0.    , 0.    , 0.    , 0.25  ]),
		np.array([1.    , 0.125 , 1.    , 0.    , 0.25  , 0.    , 0.125 , 0.    ,
       0.0625, 0.125 , 1.    , 0.0625, 0.    , 0.    , 0.    , 0.    ,
       0.25  , 0.    , 0.    , 0.    , 0.125 , 0.    , 0.25  , 0.25  ,
       1.    , 0.    , 0.    , 0.    , 1.    , 0.0625])
       ]

if __name__ == "__main__":

	if not alt_set:
		train_eids, test_eids, validate_eids = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r')), json.load(open("validate_eids", 'r'))
	else:
		train_eids, test_eids, validate_eids = json.load(open("train_eids_alt", 'r')), json.load(open("test_eids_alt", 'r')), json.load(open("validate_eids", 'r'))
	mice_data = pd.read_csv(file, low_memory=False)

	eid_to_mouse = {}
	for t_e in (train_eids + test_eids + validate_eids):
		eid_to_mouse[t_e] = mice_data[mice_data.session == t_e].subject.values[0]

	# pickle.dump(eid_to_mouse, open("eid_to_mouse.p", 'wb'))

	print("Num sessions train {}".format(len(train_eids)))
	print("Num sessions test {}".format(len(test_eids)))
	print("Num sessions validate {}".format(len(validate_eids)))
	# Num sessions train 403
	# Num sessions test 68
	# Num sessions validate 68

	all_mice = set()

	total_trials = 0
	for session in train_eids:
		temp = mice_data[mice_data.session == session]
		total_trials += len(temp)
		all_mice.add(temp.subject.values[0])
	print("Total num trials train {}".format(total_trials))
	# Total num trials train 262569

	total_trials = 0
	for session in test_eids:
		temp = mice_data[mice_data.session == session]
		total_trials += len(temp)
		all_mice.add(temp.subject.values[0])
	print("Total num trials test {}".format(total_trials))
	# Total num trials test 43278

	total_trials = 0
	for session in validate_eids:
		temp = mice_data[mice_data.session == session]
		total_trials += len(temp)
		all_mice.add(temp.subject.values[0])
	print("Total num trials validate {}".format(total_trials))
	# Total num trials validate 43968

	mice_data[['contrastLeft', 'contrastRight']] = mice_data[['contrastLeft', 'contrastRight']].fillna(0)
	mice_data['signed_contrast'] = mice_data['contrastRight'] - mice_data['contrastLeft']
	train_data = mice_data[mice_data['session'].isin(train_eids)]
	counts = train_data.groupby(['signed_contrast', 'choice']).size().unstack(fill_value=0)
	total_counts = train_data.groupby('signed_contrast').size()
	pmf = counts.divide(total_counts, axis=0) * 100
	print("Responses by contrasts")
	print(pmf)
	# Responses by contrasts
	# choice                -1.0       0.0        1.0
	# signed_contrast                                
	# -1.0000           3.500751  0.186885  96.312364
	# -0.2500           9.276771  0.313030  90.410200
	# -0.1250          18.174171  0.484555  81.341275
	# -0.0625          31.084181  0.513505  68.402314
	#  0.0000          51.965207  0.704294  47.330499
	#  0.0625          71.480651  0.508273  28.011077
	#  0.1250          83.037227  0.360373  16.602400
	#  0.2500          91.055232  0.317086   8.627682
	#  1.0000          96.589923  0.114595   3.295482

	new_df = []
	for sess in np.unique(train_data.session):
		new_df.append(train_data[train_data.session == sess].iloc[0])

	new_df = pd.DataFrame(new_df)

	counts = new_df.groupby(['signed_contrast', 'choice']).size().unstack(fill_value=0)
	total_counts_first_trial = new_df.groupby('signed_contrast').size()
	first_pmf = counts.divide(total_counts_first_trial, axis=0) * 100
	print("Responses by contrasts on trial 1")
	print(first_pmf)

	# choice                -1.0       0.0        1.0
	# signed_contrast                                
	# -1.000            5.421687  0.602410  93.975904
	# -0.250           21.739130  0.000000  78.260870
	# 0.125           71.428571  0.000000  28.571429
	# 0.250           81.818182  2.597403  15.584416
	# 1.000           94.308943  1.626016   4.065041


	left_behav = train_data[train_data.probabilityLeft == 0.8]
	right_behav = train_data[train_data.probabilityLeft == 0.2]
	neutral_behav = train_data[train_data.probabilityLeft == 0.5]

	pmf = left_behav.groupby(['signed_contrast', 'choice']).size().unstack(fill_value=0)
	plt.plot(pmf.index, pmf.values[:, 0] / pmf.values.sum(1), label="Left block", color='blue')

	pmf = right_behav.groupby(['signed_contrast', 'choice']).size().unstack(fill_value=0)
	plt.plot(pmf.index, pmf.values[:, 0] / pmf.values.sum(1), label="Right block", color='red')

	pmf = neutral_behav.groupby(['signed_contrast', 'choice']).size().unstack(fill_value=0)
	plt.plot(pmf.index, pmf.values[:, 0] / pmf.values.sum(1), label="Neutral block", color='grey')

	plt.xticks([-1, -0.25, 0, 0.25, 1], [-1, -0.25, 0, 0.25, 1])

	plt.xlim(-1, 1)
	plt.ylim(0, 1)

	plt.axhline(0.5, color='black', alpha=0.15)
	plt.axvline(0, color='black', alpha=0.15)

	plt.legend(frameon=False, fontsize=17)
	plt.xlabel('Contrast', fontsize=22)
	plt.ylabel('P(rightwards choice)', fontsize=22)

	plt.tick_params(axis='both', which='major', labelsize=17)

	sns.despine()
	plt.tight_layout()
	plt.savefig("block_pmfs.png")
	plt.close()

	print("All mice with a considered session: {}".format(len(all_mice)))
	# All mice with a considered session: 139

	input_seq, train_mask, input_seq_test, test_mask = load_data.gib_data_fast()
	shifted_input = input_seq.copy()
	shifted_input[:, :-1, :3] = shifted_input[:, 1:, :3]
	shifted_input[:, :-1, 5] = shifted_input[:, 1:, 5]

	shifted_input = shifted_input[:, :-1, [0, 1, 2, 3, 4, 5]]
	shifted_input = np.concatenate(shifted_input, axis=0)

	types, counts = np.unique(shifted_input, axis=0, return_counts=1)
	types = [tuple(t) for t in types]
	type_to_count = dict(zip(types[1:], counts[1:]))
	pickle.dump(type_to_count, open("type_to_count", 'wb'))

	def find_positions(arr, n):
		positions = []
		count = 0

		for i in range(len(arr)):
			if arr[i] == 1:
				count += 1
				if count >= n:
					positions.append(i - n + 1)
			else:
				count = 0

		return positions

	# code for finding long sequences of same answer?
	# l_total = 0
	# r_total = 0
	# for i in range(input_seq.shape[0]):
	# 	l_num = len(find_positions(input_seq[i, :, 0], 18))
	# 	r_num = len(find_positions(input_seq[i, :, 2], 18))
	# 	l_total += l_num
	# 	r_total += r_num
	# 	print("{}, Leftwards {}".format(i, l_num))
	# 	print("{}, Rightwards {}".format(i, r_num))

	# print(l_total)
	# print(r_total)

	# plot histograms of contrasts following another - contrast autocorrelation
	contrasts = input_seq[:, 1:, 3] - input_seq[:, 1:, 4]  # first row is empty, to accomodate shifts
	cont_to_num = dict(zip([-1., -0.25, -0.125, -0.0625,  0.,  0.0625,  0.125, 0.25,  1.], range(9)))
	cont_nums = np.vectorize(cont_to_num.get)(contrasts)

	occurences = np.zeros((9, 3, 9))
	for i in range(train_mask.shape[0]):
		sess = cont_nums[i][train_mask[i]]
		for j in range(sess.size - 1):
			for k in range(1, min(4, sess.size - j)):
				occurences[sess[j], k-1, sess[j + k]] += 1

	plt.figure(figsize=(16, 9))
	contrasts = [-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]
	ylabels = ["Contrasts t+1", "Contrasts t+2", "Contrasts t+3"]
	for i in range(9):
		for j in range(3):
			plt.subplot(3, 9, j * 9 + i + 1)
			plt.bar(range(9), occurences[i, j], color=['b', 'b', 'b', 'b', 'grey', 'r', 'r', 'r', 'r'])
			plt.ylim(0, 5300)
			if j == 0:
				plt.title("Contrast at t: {}".format(contrasts[i]))
			if j == 2:
				plt.gca().set_xticks([0, 2, 4, 6, 8], [-1, -0.125, 0, 0.125, 1])
			else:
				plt.gca().set_xticks([])
			if i != 0:
				plt.gca().set_yticks([])
			else:
				plt.ylabel(ylabels[j], size=18)

	sns.despine()
	plt.tight_layout()
	plt.savefig('contrast autocorr')
	plt.close()

	print("Percentage of sessions with 400 trials or lower in train: {}".format(np.mean(train_mask.sum(1) < 401)))
	print("Percentage of sessions with 400 trials or lower in test: {}".format(np.mean(test_mask.sum(1) < 401)))
	# Percentage of sessions with 400 trials or lower in train: 0.0024813895781637717
	# Percentage of sessions with 400 trials or lower in test: 0.0
	print("Percentage of sessions with less than 400 trials in train: {}".format(np.mean(train_mask.sum(1) < 400)))
	print("Percentage of sessions with less than 400 trials in test: {}".format(np.mean(test_mask.sum(1) < 400)))
	# Percentage of sessions with less than 400 trials in train: 0.0
	# Percentage of sessions with less than 400 trials in test: 0.0

	plt.hist(train_mask.sum(1))
	plt.close()

	plt.hist(test_mask.sum(1))
	plt.close()


	def count_unique_sessions(eids):
		mouse_sessions = {}
		initial_contrasts = []
		for session in eids:
			temp_data = mice_data[mice_data.session == session]
			initial_contrasts.append(np.nan_to_num(temp_data.contrastLeft.values[:30]))
			if eid_to_mouse[session] not in mouse_sessions:
				mouse_sessions[eid_to_mouse[session]] = [session]
			else:
				mouse_sessions[eid_to_mouse[session]].append(session)

		types, counts = np.unique(initial_contrasts, axis=0, return_counts=True)
		print("occurences of unique sessions: {}".format(counts))
		return types, counts, mouse_sessions

	# see how many of which unique session are in the used dataset
	print("Total ", end='')
	total_types, total_counts, mouse_sessions = count_unique_sessions(train_eids + test_eids + validate_eids)
	quit()
	# Number of unique sessions: 12
	# Total occurences of unique sessions: [  6  32  13  24 107   8  98  30   5  30  90  96]

	total_types = total_types[np.argsort(-total_counts)]
	mapping = {tuple(arr): val for arr, val in zip(total_types, list(range(12)))}
	pickle.dump(mapping, open("sess_to_num.p", 'wb'))
	quit()

	print("Train ", end='')
	train_types, train_counts, _ = count_unique_sessions(train_eids)
	print("Test ", end='')
	test_types, test_counts, _ = count_unique_sessions(test_eids)
	print("Validate ", end='')
	validate_types, validate_counts, _ = count_unique_sessions(validate_eids)
	# Train occurences of unique sessions: [ 4 25  8 20 74  6 77 21  2 22 71 73]
	# Test occurences of unique sessions: [ 3  2  3 15  2 10  8  1  3 10 11]
	# Validate occurences of unique sessions: [ 2  4  3  1 18 11  1  2  5  9 12]

	for bt in big_types:
		tot = total_counts[np.argwhere(np.all(total_types == bt, axis=1))[0, 0]]
		tra = train_counts[np.argwhere(np.all(train_types == bt, axis=1))[0, 0]]
		tes = test_counts[np.argwhere(np.all(test_types == bt, axis=1))[0, 0]]
		val = validate_counts[np.argwhere(np.all(validate_types == bt, axis=1))[0, 0]]
		print(tra / tot, tes / tot, val / tot, tot)

	# 0.6915887850467289 0.14018691588785046 0.16822429906542055 107
	# 0.7857142857142857 0.10204081632653061 0.11224489795918367 98
	# 0.7888888888888889 0.1111111111111111 0.1 90
	# 0.7604166666666666 0.11458333333333333 0.125 96



	# alt_set
	# Num sessions train 403
	# Num sessions test 68
	# Num sessions validate 68
	# Total num trials train 261304
	# Total num trials test 44543
	# Total num trials validate 43968
	# Responses by contrasts
	# choice                -1.0       0.0        1.0
	# signed_contrast                                
	# -1.0000           3.597195  0.178680  96.224125
	# -0.2500           9.364933  0.295839  90.339228
	# -0.1250          18.623619  0.499575  80.876805
	# -0.0625          31.751837  0.510679  67.737483
	#  0.0000          52.721088  0.712977  46.565934
	#  0.0625          72.204686  0.512065  27.283249
	#  0.1250          83.381945  0.367290  16.250765
	#  0.2500          91.253870  0.291169   8.454961
	#  1.0000          96.520140  0.128370   3.351490
	# All mice with a considered session: 139
	# Percentage of sessions with 400 trials or lower in train: 0.0024813895781637717
	# Percentage of sessions with 400 trials or lower in test: 0.0
	# Percentage of sessions with less than 400 trials in train: 0.0
	# Percentage of sessions with less than 400 trials in test: 0.0
	# Total occurences of unique sessions: [  6  32  13  24 107   8  98  30   5  30  90  96]
	# Train occurences of unique sessions: [ 2 27  9 22 76  7 74 22  3 24 68 69]
	# Test occurences of unique sessions: [ 2  1  1  1 13  1 13  7  1 13 15]
	# Validate occurences of unique sessions: [ 2  4  3  1 18 11  1  2  5  9 12]
	# 0.7102803738317757 0.12149532710280374 0.16822429906542055 107
	# 0.7551020408163265 0.1326530612244898 0.11224489795918367 98
	# 0.7555555555555555 0.14444444444444443 0.1 90
	# 0.71875 0.15625 0.125 96
