import numpy as np
import matplotlib.pyplot as plt
import pickle
from load_network import loaded_network
import numpy as np
import seaborn as sns
from test_code.exp_filter_tests import exp_filter_naive, exp_filter_split, exp_filter_reward, exp_filter_split_and_reward
import figrid as fg
import jax
import load_data


input_seq, train_mask, _, _ = load_data.gib_data_fast()


fig = plt.figure(figsize=(6, 9))
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0., 1], yspan=[0., 0.41]),
        'panel_B': fg.place_axes_on_grid(fig, xspan=[0, 1], yspan=[0.59, 1.])}
label_size = 24
ticksize = 18
hist_lw = 2
hist_size = 40

points = np.arange(-10, 50)
canonical_actions = input_seq[0, 291:351, 2] - input_seq[0, 291:351, 0]

_, _, test_input_seq, test_test_mask = load_data.gib_data_fast()
file_____ = 'final_net_save_88617248.p'
infos_____ = pickle.load(open("./final_nets/" + file_____, 'rb'))
reload_net_____ = loaded_network(infos_____)

assert np.allclose(100 * np.exp(-infos_____['best_test_nll']), 100 * np.exp(np.log(reload_net_____.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

normal_decay_____ = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
history_weight_____ = infos_____['best_net']['final_net']['history_weighting']

# Subplot a: Base PMF
ax_a = ax['panel_A']
pmf = reload_net_____.plot_pmf_energies(show=False)
ax_a.plot(pmf, 'k', lw=6)
ax_a.set_title("Simple PMF", size=label_size)
ax_a.axhline(0, c='k', alpha=0.2)
ax_a.axvline(4, c='k', alpha=0.2)
ax_a.set_ylim(-4.8, 4.8)
ax_a.set_xlim(0, 8)
ax_a.set_xticks([0, 2, 4, 6, 8])
ax_a.set_xticklabels([-1, -0.125, 0, 0.125, 1])
ax_a.set_ylabel("Right - left\ncontrast logit", size=label_size)
ax_a.set_xlabel("Contrast", size=label_size)
ax_a.tick_params(axis='both', which='major', labelsize=ticksize)
sns.despine(ax=ax_a)


# Subplot d: Base decay
ax_d = ax['panel_B']
ax_d.scatter(points, canonical_actions, c='k', s=hist_size, alpha=0.8, label="Choices")
filter_base = exp_filter_naive(normal_decay_____, canonical_actions)
ax_d.plot(points, filter_base * history_weight_____, alpha=0.8, c='k', lw=hist_lw, label="History component")
ax_d.axhline(0, c='k', alpha=0.2)
ax_d.legend(loc='upper left', fontsize=ticksize-4, frameon=False)
ax_d.set_title("Simple decay", size=label_size)
ax_d.set_ylim(-2, 2)
ax_d.set_xlim(0, 50)
ax_d.set_ylabel("Right - left\nhistory logit", size=label_size)
ax_d.set_xlabel("Trial", size=label_size)
ax_d.set_xticks([0, 25, 50], ['n', 'n+25', 'n+50'])
ax_d.tick_params(axis='both', which='major', labelsize=ticksize)
sns.despine(ax=ax_d)

plt.savefig("base_decay.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()