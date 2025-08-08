import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import pickle
from load_network import loaded_network
import numpy as np
import load_data
import seaborn as sns
from load_network import contrasts
import json
from scipy.stats import gaussian_kde
from test_code.exp_filter_tests import exp_filter_naive, exp_filter_split, exp_filter_reward, exp_filter_split_and_reward
import figrid as fg
import jax

contrasts = np.array(contrasts)

label_size = 20
ticksize = 18
legendsize = 13
history_lims = (-3.2, 3.2)
history_lims_sub = (-2.6, 2.6)
hist_lw = 2
hist_size = 40

# points = np.arange(-10, 51)
# canonical_actions = np.array([1, -1, -1, 1, 1, -1, -1, 1, -1, -1] + [-1., -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1., 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1])
# canonical_rewards = np.array([1, -1, -1, 1, 1, 1, 1, 1, 1, 1] + [0., 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0] + [0., 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])

trial_max = 25
points = np.arange(-20, trial_max)
canonical_actions = np.array([-1., 1, -1, 1, -1, -1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1] + 
                             [1] + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1] + [1, 1, 1, 1, 1, 1, 1, 1, 1])
alternate_actions = np.array([-1., 1, -1, 1, -1, -1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1] + 
                             [-1] + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1] + [-1, -1, 1, 1, 1, 1, 1, 1, 1])
canonical_rewards = np.ones(45)
alternate_reward = np.ones(45)
alternate_reward[25] = -1
alternate_reward[-8] = -1
alternate_reward[-9] = -1

# canonical_actions = np.ones(60)
# canonical_actions[20:40] = -1

def get_all_scalars_fast(reload_net, input, mask, infos):
    # Iterate over all sessions, saving the scalar
    all_scalar = []

    results = reload_net.lstm_fn.apply(reload_net.params, None, input[:, :, reload_net.input_list])
    if infos['agent_class'] == "<class '__main__.Simple_Infer_decay'>":
        hidden_states = results[1][0][0]  # extract the exact hidden states of the LSTM
    else:
        hidden_states = results[1][1][0]  # extract the exact hidden states of the LSTM

    if infos['agent_class'] == "<class '__main__.Mirror_mecha_plus'>":
        mot_matrix, mot_bias = infos['best_net']['mirror_mecha_plus/~_state_lstm/linear_8']['w'], infos['best_net']['mirror_mecha_plus/~_state_lstm/linear_8']['b']
    elif infos['agent_class'] == "<class '__main__.Mecha_history_plust_lstm'>":
        mot_matrix, mot_bias = infos['best_net']['mecha_history_plust_lstm/~_state_lstm/linear_8']['w'], infos['best_net']['mecha_history_plust_lstm/~_state_lstm/linear_8']['b']
    elif infos['agent_class'] == "<class '__main__.Simple_Infer_decay'>":
        mot_matrix, mot_bias = infos['best_net']['simple__infer_decay/~_state_lstm/linear_8']['w'], infos['best_net']['simple__infer_decay/~_state_lstm/linear_8']['b']
    if ('lstm_dim' not in infos or infos['lstm_dim'] == 1) and infos['agent_class'] != "<class '__main__.Mecha_history'>":
        mot_matrix = mot_matrix[:, 0]  # need to get rid of the flat last dimension for this

    for sess in range(input.shape[0]):

        if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
            motivations = (hidden_states[sess, :mask[sess].sum()] * mot_matrix).sum(1) + mot_bias
        else:
            motivations = (hidden_states[sess, :mask[sess].sum()] @ mot_matrix) + mot_bias  # TODO: here as well this clause might always work if not doing earlier if?

        all_scalar.append(-motivations + 1)

    return all_scalar

def return_exp_pmf():
    # history of best 100
    file = "mecha_sweep_indep_save_14135537.p"
    ind = 4990
    nll = 69.61903934487566

    intermediates = pickle.load(open("./best_nets_params/" + file[:-2] + "_intermediate.p", 'rb'))
    infos = pickle.load(open("./best_nets_params/" + file, 'rb'))
    if 'params_list' in infos:
        intermediates['params_list'] = infos['params_list']
    infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    reload_net_standard = loaded_network(infos, use_best=False)

    computed_nll = 100 * np.exp(np.log(reload_net_standard.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean())
    print(nll, computed_nll)
    assert np.allclose(nll, computed_nll)

    exp_pmf = reload_net_standard.plot_pmf_energies(show=False)
    return exp_pmf

training_data = True
if training_data:
    input_seq, train_mask, _, _ = load_data.gib_data_fast()
    mask_to_use = train_mask
    biased_blocks = np.load("./processed_data/train_bias.npy")
else:
    _, _, input_seq, test_mask = load_data.gib_data_fast()
    mask_to_use = test_mask
    biased_blocks = np.load("./processed_data/test_bias.npy")

_, _, test_input_seq, test_test_mask = load_data.gib_data_fast()
file="./processed_data/all_mice.csv"
train_eids, test_eids = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r'))  # map between eids and input array

gap = 0.04
hgap = 0.1
plot_width = (1 - 3 * gap) / 4
plot_height = (1 - 2 * hgap) / 3

h_correction = 0.04

fig = plt.figure(figsize=(16, 11.25))  # paper measures
# fig = plt.figure(figsize=(16, 7))  # poster measures
# ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0., plot_width], yspan=[0.2, 0.8]),
#         'panel_B': fg.place_axes_on_grid(fig, xspan=[plot_width + gap, 2 * plot_width + gap], yspan=[0., 0.47]),
#         'panel_C': fg.place_axes_on_grid(fig, xspan=[2 * plot_width + 2 * gap, 3 * plot_width + 2 * gap], yspan=[0., 0.47]),
#         'panel_D': fg.place_axes_on_grid(fig, xspan=[3 * plot_width + 3 * gap,  1.], yspan=[0., 0.47]),
#         'panel_B2': fg.place_axes_on_grid(fig, xspan=[plot_width + gap, 2 * plot_width + gap], yspan=[0.53, 1.]),
#         'panel_C2': fg.place_axes_on_grid(fig, xspan=[2 * plot_width + 2 * gap, 3 * plot_width + 2 * gap], yspan=[0.53, 1.]),
#         'panel_D2': fg.place_axes_on_grid(fig, xspan=[3 * plot_width + 3 * gap, 1.], yspan=[0.53, 1.])}
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0. + plot_width / 3 + gap, gap + plot_width * 1.6666], yspan=[0., plot_height]),
        'panel_B': fg.place_axes_on_grid(fig, xspan=[2 * plot_width + 2 * gap, 3 * plot_width + 2 * gap], yspan=[0., plot_height]),
        'panel_C': fg.place_axes_on_grid(fig, xspan=[0, plot_width], yspan=[plot_height + hgap + h_correction, 2 * plot_height + hgap + h_correction]),
        'panel_D': fg.place_axes_on_grid(fig, xspan=[0, plot_width], yspan=[2 * plot_height + 2 * hgap, 1]),
        'panel_B2': fg.place_axes_on_grid(fig, xspan=[3 * plot_width + 3 * gap, 4 * plot_width + 3 * gap], yspan=[0., plot_height]),
        'panel_C2': fg.place_axes_on_grid(fig, xspan=[plot_width + gap, 2 * plot_width + gap], yspan=[plot_height + hgap + h_correction, 2 * plot_height + hgap + h_correction]),
        'panel_D2': fg.place_axes_on_grid(fig, xspan=[plot_width + gap, 2 * plot_width + gap], yspan=[2 * plot_height + 2 * hgap, 1])}
label_size = 24
ticksize = 18

# Subplot a: Base PMF
# ax_a = fig.add_subplot(gs[0, 0])
# pmf = return_exp_pmf()
# ax_a.plot(pmf, 'k', lw=6)
# ax_a.set_title("Base PMF (SP)", size=label_size)
# ax_a.axhline(0, c='k', alpha=0.1)
# ax_a.axvline(4, c='k', alpha=0.1)
# ax_a.set_ylim(-5.8, 5.8)
# ax_a.set_xlim(0, 8)
# ax_a.set_xticks([0, 2, 4, 6, 8])
# ax_a.set_xticklabels([-1, -0.125, 0, 0.125, 1])
# ax_a.set_ylabel("Right - left logit", size=label_size)
# ax_a.set_xlabel("Contrast", size=label_size)
# ax_a.tick_params(axis='both', which='major', labelsize=ticksize)
# sns.despine(ax=ax_a)

# Subplot b: Adaptive PMF
infos = pickle.load(open("./210_twodec_rewonly_symm_noenc_regularised//simple_decay_infer_save_76678850.p", 'rb'))
reload_net = loaded_network(infos)
all_scalars = get_all_scalars_fast(reload_net, input_seq, mask_to_use, infos)
long_array = np.concatenate(all_scalars)
z_contrast = gaussian_kde(long_array)
contrast_pos = np.linspace(long_array.min(), long_array.max(), num=100)
contrast_density = z_contrast(contrast_pos)

# get the base decay and weight
file_____ = 'final_net_save_88617248.p'
infos_____ = pickle.load(open("./final_nets/" + file_____, 'rb'))
reload_net_____ = loaded_network(infos_____)

assert np.allclose(100 * np.exp(-infos_____['best_test_nll']), 100 * np.exp(np.log(reload_net_____.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
normal_decay_____ = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
history_weight_____ = infos_____['best_net']['final_net']['history_weighting']

pmf = reload_net_____.plot_pmf_energies(show=False)

augs, colors = [np.array(0.35), np.array(0.8080196062272245), np.array(1.6)], ['cyan', 'darkorange', 'indigo']

cmap = plt.get_cmap("plasma")
# Number of lines
n_lines = 7

augs = np.linspace(0.3 + 0.05, 1.65 + 0.05, n_lines)

for i, augmenter in enumerate(augs):
    augmented_contrast = np.hstack((contrasts, np.full((contrasts.shape[0], 1), augmenter)))
    pmf_energies = np.zeros(9)
    if 'symmetric_contrast_net' in infos and infos['symmetric_contrast_net']:
        for j, c in enumerate(contrasts):
            activities = reload_net.cont_process(c.reshape(1, -1), addon=augmenter.reshape(1, -1))[0]
            pmf_energies[j] = activities[0] - activities[2]
    else:
        for j, c in enumerate(augmented_contrast):
            activities = reload_net.cont_process(c)
            pmf_energies[j] = activities[0] - activities[2]
    ax['panel_A'].plot(pmf_energies, color=cmap(i / (n_lines - 1)), lw=2)

# ax['panel_A'].plot(pmf, 'k', alpha=0.8, lw=3, label="abcd")

ax['panel_A'].set_title(r"$\bf{A}$daptive PMF", size=label_size, loc='left')
ax['panel_A'].axhline(0, c='k', alpha=0.1)
ax['panel_A'].axvline(4, c='k', alpha=0.1)
ax['panel_A'].set_ylim(-5.8, 5.8)
ax['panel_A'].set_yticks([-4, 4])
ax['panel_A'].set_xlim(0, 8)
ax['panel_A'].set_xticks([0, 2, 4, 6, 8])
ax['panel_A'].set_xticklabels([-1, -0.125, 0, 0.125, 1])
ax['panel_A'].set_xlabel("Contrast", size=label_size)
ax['panel_A'].tick_params(axis='both', which='major', labelsize=ticksize)
ax['panel_A'].legend(frameon=False, fontsize=legendsize, loc='lower right')
sns.despine(ax=ax['panel_A'])

# Inset in b
ax_inset = fig.add_axes([0.23, 0.82, 0.09, 0.065])
ax_inset.plot(contrast_pos, contrast_density, 'k')
ax_inset.set_ylim(0)
ax_inset.set_ylabel("P", fontsize=ticksize-5, rotation=0)
ax_inset.set_xlabel("PMF scalar", fontsize=ticksize-7, labelpad=-1)
ax_inset.set_yticks([0, 1])
for i, augmenter in enumerate(augs):
    ax_inset.axvline(-augmenter + 1, color=cmap(i / (n_lines - 1)), alpha=0.6, lw=1.25)
sns.despine(ax=ax_inset)

# Subplot c: Two histories

file__b___ = 'final_net_save_88923355.p'
infos__b__ = pickle.load(open("./final_nets/" + file__b___, 'rb'))
reload_net__b__ = loaded_network(infos__b__)

assert np.allclose(100 * np.exp(-infos__b__['best_test_nll']), 100 * np.exp(np.log(reload_net__b__.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

higher_hist_weight = infos__b__['best_net']['final_net']['history_weighting']
lower_hist_weight = infos__b__['best_net']['final_net']['history_weighting_2']
normal_decay = jax.nn.sigmoid(reload_net__b__.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
slow_decay = jax.nn.sigmoid(reload_net__b__.final_decay_2(np.array(0).reshape(1, -1)))[0, 0]

ax['panel_B'].plot(1 + points, exp_filter_naive(normal_decay, canonical_actions) * higher_hist_weight, label='Short history', alpha=0.8, c='sandybrown', lw=hist_lw)
ax['panel_B'].plot(1 + points, exp_filter_naive(slow_decay, canonical_actions) * lower_hist_weight, label='Long history', alpha=0.8, c='saddlebrown', lw=hist_lw)
ax['panel_B'].scatter(points, canonical_actions, c='k', s=hist_size, alpha=0.8, label='Choices')
ax['panel_B'].axhline(0, c='k', alpha=0.1)
ax['panel_B'].set_title(r"$\bf{B}$i histories", size=label_size, loc='left')
ax['panel_B'].set_ylim(*history_lims)
ax['panel_B'].set_yticks([-2, 0, 2], [-2, 0, 2])
ax['panel_B'].set_xlim(0, trial_max)
ax['panel_B'].legend(frameon=False, fontsize=legendsize, loc='upper left')
# ax['panel_B'].set_xlabel("Trial", size=label_size)
sns.despine(ax=ax['panel_B'])


ax['panel_B2'].scatter(points, canonical_actions, c='k', s=hist_size, alpha=0.8)


filter_base = exp_filter_naive(normal_decay_____, canonical_actions)
# ax['panel_B2'].plot(1 + points, exp_filter_naive(normal_decay, canonical_actions) * higher_hist_weight + exp_filter_naive(slow_decay, canonical_actions) * lower_hist_weight - filter_total * history_weight_____, label='Short', alpha=0.8, c='r', lw=hist_lw)
summed__b__ = exp_filter_naive(normal_decay, canonical_actions) * higher_hist_weight + exp_filter_naive(slow_decay, canonical_actions) * lower_hist_weight
ax['panel_B2'].plot(1 + points, filter_base * history_weight_____, 'k', label='abcd', alpha=0.5, ls='-.', lw=hist_lw * 0.75)
ax['panel_B2'].plot(1 + points, summed__b__, 'k', label='aBcd', alpha=0.8, lw=hist_lw)

# for i, yi1, yi2 in zip(points, summed__b__, filter_base * history_weight_____):
#     ax['panel_B2'].plot([i+1, i+1], [yi1, yi2], color='red', linestyle='-', linewidth=1)

ax['panel_B2'].legend(frameon=False, fontsize=legendsize, loc='upper left')
ax['panel_B2'].axhline(0, c='k', alpha=0.1)
ax['panel_B2'].set_ylim(*history_lims_sub)
ax['panel_B2'].set_xlim(0, trial_max)
sns.despine(ax=ax['panel_B2'])

# Subplot d: Base decay
# ax_d = fig.add_subplot(gs[1, 0])
# ax_d.scatter(points, canonical_actions, c='k', s=hist_size, alpha=0.8, label="Choices")
# filter_base = exp_filter_naive(normal_decay_____, canonical_actions)
# ax_d.plot(filter_base * history_weight_____, alpha=0.8, c='k', lw=hist_lw, label="History component")
# ax_d.axhline(0, c='k', alpha=0.1)
# ax_d.legend(loc='lower left', fontsize=ticksize, frameon=False, bbox_to_anchor=(0.06, 0.0745), bbox_transform=fig.transFigure)
# ax_d.set_title("Base decay (1S)", size=label_size)
# ax_d.set_ylim(*history_lims)
# ax_d.set_xlim(0, trial_max)
# ax_d.set_ylabel("Right - left logit", size=label_size)
# ax_d.set_xlabel("Trial", size=label_size)
# ax_d.set_xticks([0, 20], ['n', 'n+20'])
# ax_d.tick_params(axis='both', which='major', labelsize=ticksize)
# sns.despine(ax=ax_d)

# Subplot e: Split actions


file___c_ = 'final_net_save_53997952.p'
infos__c_ = pickle.load(open("./final_nets/" + file___c_, 'rb'))
reload_net__c_ = loaded_network(infos__c_)

assert np.allclose(100 * np.exp(-infos__c_['best_test_nll']), 100 * np.exp(np.log(reload_net__c_.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

hist_weight = infos__c_['best_net']['final_net']['history_weighting']
chosen_decay, _, unchosen_decay = jax.nn.sigmoid(reload_net__c_.final_decay_1(np.array([1, 0, 0]).reshape(1, -1)))[0]

ax['panel_C'].scatter(points[canonical_actions == 1], canonical_actions[canonical_actions == 1], c='r', s=hist_size, alpha=0.8)
ax['panel_C'].scatter(points[canonical_actions == -1], canonical_actions[canonical_actions == -1], c='b', s=hist_size, alpha=0.8)
f_total, f_left, f_right = exp_filter_split(chosen_decay, unchosen_decay, canonical_actions, return_both=True)
ax['panel_C'].plot(1 + points, f_right * hist_weight, alpha=0.8, c='red', lw=hist_lw, label='Right logit')
ax['panel_C'].plot(1 + points, f_left * hist_weight, alpha=0.8, c='blue', lw=hist_lw, label='Left logit')
# ax['panel_C'].plot(1 + points, f_total * hist_weight, alpha=0.8, c='k', lw=hist_lw)
ax['panel_C'].axhline(0, c='k', alpha=0.1)
ax['panel_C'].set_title(r"$\bf{C}$hoice separation", size=label_size, loc='left')
ax['panel_C'].set_ylim(*history_lims)
ax['panel_C'].set_xlim(0, trial_max)
ax['panel_C'].legend(frameon=False, fontsize=legendsize, loc='upper left')
sns.despine(ax=ax['panel_C'])



ax['panel_C2'].scatter(points[canonical_actions == 1], canonical_actions[canonical_actions == 1], c='r', s=hist_size, alpha=0.8)
ax['panel_C2'].scatter(points[canonical_actions == -1], canonical_actions[canonical_actions == -1], c='b', s=hist_size, alpha=0.8)
ax['panel_C2'].plot(1 + points, filter_base * history_weight_____, 'k', alpha=0.5, ls='-.', lw=hist_lw * 0.75, label='abcd')
ax['panel_C2'].plot(1 + points, f_total * hist_weight, alpha=0.8, c='k', lw=hist_lw, label="abCd")

# for i, yi1, yi2 in zip(points, f_total * hist_weight, filter_base * history_weight_____):
#     ax['panel_C2'].plot([i+1, i+1], [yi1, yi2], color='red', linestyle='-', linewidth=1)

ax['panel_C2'].axhline(0, c='k', alpha=0.1)
ax['panel_C2'].set_ylim(*history_lims_sub)
ax['panel_C2'].set_xlim(0, trial_max)
ax['panel_C2'].legend(frameon=False, fontsize=legendsize, loc='upper left')
sns.despine(ax=ax['panel_C2'])


# ax['panel_C2'].scatter(points[canonical_actions == 1], canonical_actions[canonical_actions == 1], c='r', s=hist_size, alpha=0.8)
# ax['panel_C2'].scatter(points[canonical_actions == -1], canonical_actions[canonical_actions == -1], c='b', s=hist_size, alpha=0.8)
# f_total, f_left, f_right = exp_filter_split(0.50571384, 0.7595863, canonical_actions, return_both=True)
# ax['panel_C2'].plot(1 + points, f_total * 0.9 - filter_total * history_weight_____, alpha=0.8, c='r', lw=hist_lw)
# ax['panel_C2'].axhline(0, c='k', alpha=0.1)
# ax['panel_C2'].set_xticks([0, 20], ['n', 'n+20'])
# ax['panel_C2'].set_ylim(*history_lims)
# ax['panel_C2'].set_xlim(0, trial_max)
# ax['panel_C2'].set_xlabel("Trial", size=label_size)
# ax['panel_C2'].tick_params(axis='both', which='major', labelsize=ticksize)
# sns.despine(ax=ax['panel_C2'])

# Subplot f: Reward-dependent decay

file___cd = 'final_net_save_57392066.p'
infos__cd = pickle.load(open("./final_nets/" + file___cd, 'rb'))
reload_net__cd = loaded_network(infos__cd)

assert np.allclose(100 * np.exp(-infos__cd['best_test_nll']), 100 * np.exp(np.log(reload_net__cd.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

hist_weight = infos__cd['best_net']['final_net']['history_weighting']
rewarded_chosen_decay, _, rewarded_unchosen_decay = jax.nn.sigmoid(reload_net__cd.final_decay_1(np.array([1, 0, 0]).reshape(1, -1), addon=np.array([1]).reshape(1, -1)))[0]
unrewarded_chosen_decay, _, unrewarded_unchosen_decay = jax.nn.sigmoid(reload_net__cd.final_decay_1(np.array([1, 0, 0]).reshape(1, -1), addon=np.array([-1]).reshape(1, -1)))[0]

ax['panel_D'].scatter(points[alternate_reward == 1], alternate_actions[alternate_reward == 1], c='g', s=hist_size, alpha=0.8, label="Correct")
ax['panel_D'].scatter(points[alternate_reward == -1], alternate_actions[alternate_reward == -1], c='fuchsia', s=hist_size, alpha=0.8, label="Incorrect")

# ax['panel_D'].scatter([5], [-1], c='fuchsia', s=hist_size, alpha=0.8)


filter___cd, filter___cd_left, filter___cd_right = exp_filter_split_and_reward(rewarded_chosen_decay, rewarded_unchosen_decay, unrewarded_chosen_decay, unrewarded_unchosen_decay, alternate_actions, alternate_reward, return_both=True)
filter_unreward_cd, filter_unreward_cd_left, filter_unreward_cd_right = exp_filter_split_and_reward(rewarded_chosen_decay, rewarded_unchosen_decay, unrewarded_chosen_decay, unrewarded_unchosen_decay, alternate_actions, alternate_reward, return_both=True)

from matplotlib.collections import LineCollection

def add_mixed_line(x, y, colors, axis):
    new_points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([new_points[:-1], new_points[1:]], axis=1)

    # Create a LineCollection with alternating colors
    # colors = np.array([[0, 0.8, 0, 1], [1, 0, 1, 1]])  # Green and Fuchsia
    # colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])  # Red and Blue
    line_segments = LineCollection(segments, colors=np.tile(colors, (len(segments)//2 + 1, 1))[:len(segments)])
    axis.add_collection(line_segments)

# add_mixed_line(1 + points[24:], filter___cd_right[24:] * hist_weight, colors=np.array([[1, 0, 0, 1], [0, 0.8, 0, 1]]), axis=ax['panel_D'])
# add_mixed_line(1 + points[24:], filter___cd_left[24:] * hist_weight, colors=np.array([[0, 0, 1, 1], [0, 0.8, 0, 1]]), axis=ax['panel_D'])
ax['panel_D'].plot(1 + points, filter_unreward_cd_right * hist_weight, c='red')
ax['panel_D'].plot(1 + points, filter_unreward_cd_left * hist_weight, c='blue')
# add_mixed_line(1 + points[24:], filter_unreward_cd_right[24:] * hist_weight, colors=np.array([[1, 0, 0, 1], [1, 0, 1, 1]]), axis=ax['panel_D'])
# add_mixed_line(1 + points[24:], filter_unreward_cd_left[24:] * hist_weight, colors=np.array([[0, 0, 1, 1], [1, 0, 1, 1]]), axis=ax['panel_D'])

# box around alternate choices
# from matplotlib.patches import FancyBboxPatch
# box = FancyBboxPatch(
#     (5 - 0.25, -0.97),           # (x, y)
#     0.5,              # width
#     2 * 0.97,              # height
#     boxstyle="round,pad=0.2", # Rounded corners
#     edgecolor="k",
#     facecolor="none",
#     linewidth=2
# )
# ax['panel_D'].add_patch(box)
# draw arrows between the two choices on trial 5
# ax['panel_D'].annotate('', xy=(5, -0.97), xytext=(5, 0.97),
#             arrowprops=dict(arrowstyle="<->", lw=2, color='k'),
#             fontsize=20, ha='center', va='center')
# ax['panel_D2'].annotate('', xy=(5, -0.97), xytext=(5, 0.97),
#             arrowprops=dict(arrowstyle="<->", lw=2, color='k'),
#             fontsize=20, ha='center', va='center')

ax['panel_D'].axhline(0, c='k', alpha=0.1)
ax['panel_D'].set_title(r"$\bf{D}$ecay | reward", size=label_size, loc='left')
ax['panel_D'].set_ylim((-3.5, 3.5))
ax['panel_D2'].set_ylim((-3, 3))
ax['panel_D'].set_xlim(0, trial_max)
sns.despine(ax=ax['panel_D'])
ax['panel_D2'].scatter(points[alternate_reward == 1], alternate_actions[alternate_reward == 1], c='g', s=hist_size, alpha=0.8, label="Correct")
ax['panel_D2'].scatter(points[alternate_reward == -1], alternate_actions[alternate_reward == -1], c='fuchsia', s=hist_size, alpha=0.8, label="Incorrect")
# ax['panel_D2'].scatter([5], [-1], c='fuchsia', s=hist_size, alpha=0.8)

alternate_filter_base = exp_filter_naive(normal_decay_____, alternate_actions)
# ax['panel_D2'].plot(1 + points[:25], filter_base[:25] * history_weight_____, 'k', alpha=0.4, lw=hist_lw * 0.75)
ax['panel_D2'].plot(1 + points, alternate_filter_base * history_weight_____, 'k', alpha=0.5, ls='-.', lw=hist_lw * 0.75, label='abcd')
ax['panel_D2'].plot(1 + points, filter___cd * hist_weight, alpha=0.8, c='k', lw=hist_lw, label='abCD')
# ax['panel_D2'].plot(1 + points, filter_unreward_cd * hist_weight, alpha=0.8, c='fuchsia', lw=hist_lw)
# ax['panel_D2'].plot(1 + points, filter___cd * hist_weight, alpha=0.8, c='green', lw=hist_lw)

# add some mixed lines
# add_mixed_line(1 + points[24:], filter___cd[24:] * hist_weight, colors=np.array([[0, 0, 0, 1], [0, 0.8, 0, 1]]), axis=ax['panel_D2'])
# add_mixed_line(1 + points[24:], filter_unreward_cd[24:] * hist_weight, colors=np.array([[0, 0, 0, 1], [1, 0, 1, 1]]), axis=ax['panel_D2'])
# add_mixed_line(1 + points[24:], filter_base[24:] * history_weight_____, colors=np.array([[0, 0, 0, 0.5], [0, 0.8, 0, 1]]), axis=ax['panel_D2'])
# add_mixed_line(1 + points[24:], alternate_filter_base[24:] * history_weight_____, colors=np.array([[0, 0, 0, 0.5], [1, 0, 1, 1]]), axis=ax['panel_D2'])

# for i, yi1, yi2 in zip(points, filter___cd * hist_weight, filter_base * history_weight_____):
#     ax['panel_D2'].plot([i+1, i+1], [yi1, yi2], color='red', linestyle='-', linewidth=1)

ax['panel_D2'].axhline(0, c='k', alpha=0.1)
ax['panel_D2'].set_xlim(0, trial_max)
ax['panel_D2'].legend(loc='upper left', fontsize=legendsize, frameon=False, ncols=2)#, bbox_to_anchor=(0.7, 0.0745), bbox_transform=fig.transFigure)
sns.despine(ax=ax['panel_D2'])


ax['panel_A'].set_ylabel(r"$\Delta$ cont. logits", fontsize=label_size)

ax['panel_B'].set_ylabel(r"$\Delta$ hist. logits", fontsize=label_size)
ax['panel_B'].set_xlabel("Trial", size=label_size)
ax['panel_B'].set_xticks([0, 20], ['n', 'n+20'])
ax['panel_B'].tick_params(axis='both', which='major', labelsize=ticksize)

ax['panel_B2'].set_xlabel("Trial", size=label_size)
ax['panel_B2'].set_xticks([0, 20], ['n', 'n+20'])
ax['panel_B2'].tick_params(axis='both', which='major', labelsize=ticksize)

ax['panel_C'].set_ylabel(r"$\Delta$ hist. logits", fontsize=label_size)
ax['panel_C'].set_xticks([])
ax['panel_C'].tick_params(axis='both', which='major', labelsize=ticksize)

ax['panel_C2'].set_xticks([])
ax['panel_C2'].tick_params(axis='both', which='major', labelsize=ticksize)

ax['panel_D'].set_ylabel(r"$\Delta$ hist. logits", fontsize=label_size)
ax['panel_D'].set_xlabel("Trial", size=label_size)
ax['panel_D'].set_xticks([0, 20], ['n', 'n+20'])
ax['panel_D'].tick_params(axis='both', which='major', labelsize=ticksize)

ax['panel_D2'].set_xlabel("Trial", size=label_size)
ax['panel_D2'].set_xticks([0, 20], ['n', 'n+20'])
ax['panel_D2'].tick_params(axis='both', which='major', labelsize=ticksize)

# Panel labels
labels = [
    {'label_text': r"$\boldsymbol{\Omega}$", 'xpos': 0.515, 'ypos': 0.58}
]
for lbl in labels:
    fig.text(lbl['xpos'], lbl['ypos'], lbl['label_text'], fontsize=label_size, fontweight='bold')

plt.tight_layout()
plt.savefig("model_components.png", dpi=300, bbox_inches='tight')
plt.show()

