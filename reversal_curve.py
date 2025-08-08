import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from load_network import loaded_network
import load_data
import seaborn as sns
from scipy.stats import sem

dpi = None

plot_0_only = True

label_size = 28
ticksize = 24


yticks = []

fig = plt.figure(figsize=(8.5, 6.75))
a1 = plt.gca()

training_data = False
if training_data:
    input_seq, train_mask, _, _ = load_data.gib_data_fast()
    mask_to_use = train_mask
    biased_blocks = np.load("./processed_data/train_bias.npy")
else:
    _, _, input_seq, test_mask = load_data.gib_data_fast()
    mask_to_use = test_mask
    biased_blocks = np.load("./processed_data/test_bias.npy")

_, _, test_input_seq, test_test_mask = load_data.gib_data_fast()



bias_diff = np.abs(biased_blocks[:, :-1] - biased_blocks[:, 1:])
before, after = 8, 20
specific_contrast = [0, 0.0625, 0.125, 0.25, 1][0]



session_end = False
n_from_end = 50

performance_mice = []
performance_mice_0 = [[] for _ in range(before + after)]
# block switch plots for only 0 contrasts

for i, session_mask in enumerate(mask_to_use):
    block_switches = np.where(np.isclose(bias_diff[i], 0.6))[0]
    print(i)
    for spot in block_switches:
        spot = spot + 1  # actualy block switch is one trial later
        if session_mask[spot - before:spot + after].sum() != before + after:
            continue

        performance_mice.append((input_seq[i, spot - before + 1:spot + after + 1, 5] + 1) / 2)  # +1 since rewards, which we are pulling, are delayed one entry
        for j in range(before + after):
            if input_seq[i, spot - before + j, 3] == 0 and input_seq[i, spot - before + j, 4] == 0:
                performance_mice_0[j].append((input_seq[i, spot - before + j + 1, 5] + 1) / 2)  # +1 since rewards, which we are pulling, are delayed one entry



performance_mice = np.array(performance_mice)
if not plot_0_only:
    a1.plot(np.arange(-before, after), performance_mice.mean(0), color='k')
    a1.fill_between(np.arange(-before, after), performance_mice.mean(0) - sem(performance_mice, axis=0) / 2, performance_mice.mean(0) + sem(performance_mice, axis=0) / 2, color='k', alpha=0.2)
else:
    means = [np.mean(performance_mice_0[i]) for i in range(before + after)]
    sems = [sem(performance_mice_0[i]) for i in range(before + after)]
    # a1.plot(np.arange(-before, after), means, color='k')
    # a1.fill_between(np.arange(-before, after), np.array(means) - np.array(sems), np.array(means) + np.array(sems), color='k', alpha=0.2)
    a1.errorbar(np.arange(-before, after), means, np.array(sems), color='green')
a1.axvline(- 0.5, c='k', ls='--')  # put line just before the first trial of new block
a1.set_xlim(-before, after)
a1.set_xlabel("Trials around block switch", size=label_size)
a1.set_ylabel("Reward rate on 0% contrast", size=label_size)
a1.tick_params(axis='both', which='major', labelsize=ticksize)

a1.set_xticks([-5, 0, 5, 10, 15, 20], [-5, 0, 5, 10, 15, 20])

if not plot_0_only:
    a1.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])

# if not plot_0_only:
#     a1.set_ylim(0.7, 0.9)
#     a1.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])
# else:
#     # a2.set_ylim(0.3, 0.75)
#     pass

sns.despine()


plt.tight_layout()

plt.savefig(f"reversal_0_only_{plot_0_only}", dpi=250)
plt.show()
