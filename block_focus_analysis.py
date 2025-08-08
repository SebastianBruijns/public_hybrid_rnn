import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from load_network import loaded_network
import load_data
import seaborn as sns
from scipy.stats import sem

model_performance_color = '#4a0100'

training_data = False
if training_data:
    input_seq, train_mask, _, _ = load_data.gib_data_fast()
    mask_to_use = train_mask
    biased_blocks = np.load("./processed_data/train_bias.npy")
else:
    _, _, input_seq, test_mask = load_data.gib_data_fast()
    mask_to_use = test_mask
_, _, test_input_seq, test_test_mask = load_data.gib_data_fast()

eval_on_heldout = True

if eval_on_heldout:
    input_seq, mask_to_use, biased_blocks = load_data.gib_test_data()

# best ----
file_____ = 'final_net_save_88617248.p'
infos_____ = pickle.load(open("./final_nets/" + file_____, 'rb'))
reload_net_____ = loaded_network(infos_____)

print(100 * np.exp(-infos_____['best_test_nll']), 100 * np.exp(np.log(reload_net_____.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
assert np.allclose(100 * np.exp(-infos_____['best_test_nll']), 100 * np.exp(np.log(reload_net_____.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

# best abcd
file_abcd = 'final_net_save_34114033.p'
infos_abcd = pickle.load(open("./final_nets/" + file_abcd, 'rb'))
reload_net_abcd = loaded_network(infos_abcd)

print(100 * np.exp(-infos_abcd['best_test_nll']), 100 * np.exp(np.log(reload_net_abcd.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
assert np.allclose(100 * np.exp(-infos_abcd['best_test_nll']), 100 * np.exp(np.log(reload_net_abcd.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

predictions_____ = reload_net_____.return_predictions(input_seq, action_subsetting=True)
predictions_abcd = reload_net_abcd.return_predictions(input_seq, action_subsetting=True)

performance = predictions_abcd - predictions_____

bias_diff = np.abs(biased_blocks[:, :-1] - biased_blocks[:, 1:])
before, after = 8, 20

performance_period = []
for i, session_mask in enumerate(mask_to_use):
    block_switches = np.where(np.isclose(bias_diff[i], 0.6))[0]
    for spot in block_switches:
        spot = spot + 1  # actualy block switch is one trial later
        if session_mask[spot - before:spot + after].sum() != before + after:
            continue
        performance_period.append(performance[i, spot - before:spot + after])

performance_period = np.array(performance_period)

n_trials_to_average = performance_period.shape[0]
n_permute = 100000

permuted_means = np.random.choice(performance[mask_to_use], size=(n_permute, n_trials_to_average), replace=True)

plt.figure(figsize=(13, 7))

# plt.plot(np.arange(-before, after), performance_period.mean(0), label="ABCD - abcd", lw=2.5, c=model_performance_color)
plt.errorbar(np.arange(-before, after), performance_period.mean(0), sem(performance_period, axis=0), label="ABCD - abcd", lw=2.5, c=model_performance_color)
a, b, c = np.percentile(permuted_means.mean(1), [97.5, 99.5, 99.95])
plt.axhline(a, color='g', label="p=0.05")
plt.axhline(b, color='r', label="p=0.01")
plt.axhline(c, color='gold', label="p=0.001")

plt.axvline(-0.5, color='k', alpha=0.3)

plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.legend(frameon=False, fontsize=20)
plt.xlim(-before, after)

plt.xlabel("Trials around block switch", fontsize=20)
plt.ylabel("Model advantage on test", fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig("block_switch_significance")
plt.show()