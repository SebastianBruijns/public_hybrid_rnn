"""
    Extracts of the code from network_ana_1d_latent.py, to extract relevant info about LSTM scalar and how it influences the behaviour of the nets
"""
import pickle
from load_network import loaded_network
import numpy as np
import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import gaussian_kde
import imageio
from load_network import contrasts
from scipy.stats import sem
import matplotlib
import jax
import figrid as fg
# matplotlib.use('agg')

contrasts = np.array(contrasts)

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


# if this is the main file
if __name__ == "__main__":
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
    _, _, _, _, _, _, train_corrects = load_data.gib_data(return_corrects=True)


    train_eids, test_eids = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r'))  # map between eids and input array

    file = "final_net_save_88617248.p"
    infos = pickle.load(open("./final_nets/" + file, 'rb'))
    reload_net_standard = loaded_network(infos)
    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net_standard.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

    file = "simple_decay_infer_save_76678850.p"
    infos = pickle.load(open("./210_twodec_rewonly_symm_noenc_regularised//" + file, 'rb'))
    reload_net = loaded_network(infos)
    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))


    n_bounds = 4
    timeout_sessions = 0
    timeout_sessions_valid = 0
    before, after = 15, 16
    if True:
        all_scalars = get_all_scalars_fast(reload_net, input_seq, mask_to_use, infos)
        long_array = np.concatenate(all_scalars)

        bounds = np.quantile(long_array, np.linspace(0, 1, n_bounds+1))

        all_scalar = []
        all_history_comps = []
        l_comps = []
        r_comps = []
        neutral_comps = []

        interpolated_timeout_scalars = np.zeros((403, 100))
        zoomed_timeout_scalars = [[] for _ in range(1 + before + after)]

        interpolated_rewards = np.zeros((403, 100))
        zoomed_rewards = [[] for _ in range(1 + before + after)]
        zoomed_reward_estimates = [[] for _ in range(1 + before + after)]
        interpolated_reward_estimates = np.zeros((403, 100))

        results_standard = reload_net_standard.lstm_fn.apply(reload_net_standard.params, None, input_seq[:, :, reload_net_standard.input_list])
        results = reload_net.lstm_fn.apply(reload_net.params, None, input_seq[:, :, reload_net.input_list])
        hidden_states = results[1][0][0]  # extract the exact hidden states of the LSTM
        history_safe = results[1][1]
        predictions = results[0]

        choices = []

        scalar_conditioned_choices = {n: {c: [] for c in [-1., -0.25, -0.125, -0.0625, 0., 0.0625, 0.125, 0.25, 1.]} for n in range(n_bounds)}
        scalar_conditioned_best_prob = {n: {c: [] for c in [-1., -0.25, -0.125, -0.0625, 0., 0.0625, 0.125, 0.25, 1.]} for n in range(n_bounds)}
        scalar_conditioned_100_prob = {n: {c: [] for c in [-1., -0.25, -0.125, -0.0625, 0., 0.0625, 0.125, 0.25, 1.]} for n in range(n_bounds)}

        for sess in range(len(train_eids)):

            print(sess)

            choices.append(input_seq[sess, :train_mask[sess].sum()+1, 0] - input_seq[sess, :train_mask[sess].sum()+1, 2])

            choice = input_seq[sess, 1:train_mask[sess].sum()+1, 0] - input_seq[sess, 1:train_mask[sess].sum()+1, 2]
            contrasts = input_seq[sess, :train_mask[sess].sum(), 3] - input_seq[sess, :train_mask[sess].sum(), 4]

            rewards = (input_seq[sess, 1:1+train_mask[sess].sum(), 5] + 1) / 2

            # interpolate the contrast scalar
            interpolated_timeout_scalars[sess] = np.interp(np.linspace(0, len(all_scalars[sess]), 100), np.arange(len(all_scalars[sess])), all_scalars[sess])
            interpolated_rewards[sess] = np.interp(np.linspace(0, len(all_scalars[sess]), 100), np.arange(len(all_scalars[sess])), rewards)
            interpolated_reward_estimates[sess] = np.interp(np.linspace(0, len(all_scalars[sess]), 100), np.arange(len(all_scalars[sess])), predictions[sess, np.arange(len(all_scalars[sess])), (train_corrects[sess, :train_mask[sess].sum()] + 2) % 3])

            if (choice == 0).any():
                first_timeout = np.where(choice == 0)[0][0]
                timeout_sessions += 1

                base = all_scalars[sess][max(0, first_timeout - before): min(train_mask[sess].sum() - 1, first_timeout + after + 1)][0]
                base = 0
                for i, trial in enumerate(range(-before, after + 1)):
                    if 0 < first_timeout + trial < train_mask[sess].sum():
                        zoomed_timeout_scalars[i].append(all_scalars[sess][first_timeout + trial] - base)
                        zoomed_rewards[i].append(rewards[first_timeout + trial])

                        zoomed_reward_estimates[i].append(predictions[sess, first_timeout + trial, (train_corrects[sess, first_timeout + trial] + 2) % 3])

            # we will collect psychometric functions for the different levels of contrast scalar
            for i in range(train_mask[sess].sum()):
                # depending on where in bounds the scalar is, we will collect the choice
                for j in range(n_bounds):
                    if bounds[j] <= all_scalars[sess][i] < bounds[j+1]:
                        scalar_conditioned_choices[j][contrasts[i]].append(1 - (choice[i] + 1) / 2)
                        scalar_conditioned_best_prob[j][contrasts[i]].append(results[0][sess, i, 2])
                        scalar_conditioned_100_prob[j][contrasts[i]].append(results_standard[0][sess, i, 2])

                        break
            
        print(f"Timeout sessions: {timeout_sessions}")
        # Timeout sessions: 194
        pickle.dump((scalar_conditioned_choices, scalar_conditioned_best_prob, scalar_conditioned_100_prob, all_scalars, interpolated_timeout_scalars, zoomed_timeout_scalars, interpolated_rewards, zoomed_rewards, zoomed_reward_estimates, interpolated_reward_estimates), open("final_scalar_conditioned_choices_no_baseline.p", 'wb'))
    scalar_conditioned_choices, scalar_conditioned_best_prob, scalar_conditioned_100_prob, all_scalars, interpolated_timeout_scalars, zoomed_timeout_scalars, interpolated_rewards, zoomed_rewards, zoomed_reward_estimates, interpolated_reward_estimates = pickle.load(open("final_scalar_conditioned_choices_no_baseline.p", 'rb'))
    long_array = np.concatenate(all_scalars)

    # plot the psychometric functions
    fig = plt.figure(figsize=(16, 8))
    a_till, a_gap = 0.58, 0.01
    plot_width = (a_till - 3 * a_gap) / 4
    axes = {'panel_A1': fg.place_axes_on_grid(fig, xspan=[0., plot_width], yspan=[0., 0.45]),
          'panel_A2': fg.place_axes_on_grid(fig, xspan=[plot_width + a_gap, 2 * plot_width + a_gap], yspan=[0., 0.45]),
          'panel_A3': fg.place_axes_on_grid(fig, xspan=[2 * plot_width + 2 * a_gap, 3 * plot_width + 2 * a_gap], yspan=[0., 0.45]),
          'panel_A4': fg.place_axes_on_grid(fig, xspan=[3 * plot_width + 3 * a_gap,  a_till], yspan=[0., 0.45]),

          'panel_B1': fg.place_axes_on_grid(fig, xspan=[0., 0.245], yspan=[0.57, 1]),
          'panel_B2': fg.place_axes_on_grid(fig, xspan=[0.265, 0.51], yspan=[0.57, 1]),
          
          'panel_C_main': fg.place_axes_on_grid(fig, xspan=[0.695, 0.97], yspan=[0.06, 1]),
          'panel_C_marginal': fg.place_axes_on_grid(fig, xspan=[0.695, 0.97], yspan=[0., 0.05]),
          'panel_C_colorbar': fg.place_axes_on_grid(fig, xspan=[0.975, 1], yspan=[0.06, 1]),
          'panel_C_reward': fg.place_axes_on_grid(fig, xspan=[0.64, 0.69], yspan=[0.06, 1])}


    for i, ax in enumerate([axes['panel_A1'], axes['panel_A2'], axes['panel_A3'], axes['panel_A4']]):
        # ax.plot([np.mean(scalar_conditioned_choices[i][x]) for x in scalar_conditioned_choices[i]], alpha=0.66, ls='--', color='lime', lw=3, label="Empirical")
        # ax.plot([np.mean(scalar_conditioned_best_prob[i][x]) for x in scalar_conditioned_best_prob[i]], alpha=0.55, color='red', lw=3, label='ABCD')
        # ax.plot([np.mean(scalar_conditioned_100_prob[i][x]) for x in scalar_conditioned_best_prob[i]], color='k', alpha=0.33, lw=2, label='abcd')
        print("number of trials per PMF datapoint")
        print([len(scalar_conditioned_choices[i][x]) for x in scalar_conditioned_choices[i]])
        print([len(scalar_conditioned_best_prob[i][x]) for x in scalar_conditioned_best_prob[i]])
        print(min([len(scalar_conditioned_choices[i][x]) for x in scalar_conditioned_choices[i]]))
        print()
        ax.errorbar(np.arange(9), [np.mean(scalar_conditioned_choices[i][x]) for x in scalar_conditioned_choices[i]], [sem(scalar_conditioned_choices[i][x]) for x in scalar_conditioned_choices[i]], alpha=0.66, ls='--', color='lime', lw=3, label="Empirical")
        ax.errorbar(np.arange(9), [np.mean(scalar_conditioned_best_prob[i][x]) for x in scalar_conditioned_best_prob[i]], [sem(scalar_conditioned_best_prob[i][x]) for x in scalar_conditioned_best_prob[i]], alpha=0.55, color='red', lw=3, label='ABCD')
        ax.errorbar(np.arange(9), [np.mean(scalar_conditioned_100_prob[i][x]) for x in scalar_conditioned_best_prob[i]], [sem(scalar_conditioned_100_prob[i][x]) for x in scalar_conditioned_best_prob[i]], color='k', alpha=0.33, lw=2, label='abcd')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 8)

        if i == 0:
            ax.legend(frameon=False, fontsize=12, loc=2)

        ax.axvline(4, color='k', alpha=0.15)
        ax.axhline(0.5, color='k', alpha=0.15)
        ax.axvline(4, color='k', alpha=0.15)
        ax.axhline(0.5, color='k', alpha=0.15)

        ax.set_title(rf"{i+1}. $s^c$ Quartile", fontsize=18)
        if i != 0:
            ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [None, None, None, None, None, None, None, None, None])
            # axs[i // 4, i % 4].set_xticks([])
            # axs[i // 4, i % 4].set_yticks([])
        else:
            ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [-1, None, None, None, 0, None, None, None, 1])
            ax.set_xlabel("Contrast", fontsize=17)
            ax.set_ylabel("P(choose rightwards)", fontsize=17)
            
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [None, None, None, None, None])

    axes['panel_A1'].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
    


    avg_rewards = []
    for sess in range(input_seq.shape[0]):
        mask = mask_to_use[sess]
        avg_rewards.append(np.mean(np.clip(input_seq[sess, :-1, 5][mask], 0, 1)))

    points = np.linspace(long_array.min(), long_array.max(), 200)
    session_marginals = np.zeros((len(all_scalars), 200))


    for i in range(len(all_scalars)):
        density = gaussian_kde(all_scalars[i])

        session_marginals[i] = density(points)

    session_marginals = session_marginals[np.argsort(avg_rewards)[::-1]]

    # add a column above the plot, showing the overall scalar density
    density = gaussian_kde(long_array)
    overall_density = density(points)


    im0 = axes['panel_C_marginal'].imshow(overall_density.reshape(1, -1), cmap='viridis', aspect='auto')
    axes['panel_C_marginal'].set_title("Marginal over sessions", size=17)
    
    im1 = axes['panel_C_main'].imshow(session_marginals, aspect='auto', cmap='viridis')

    axes['panel_C_main'].set_ylim(402.5, -0.5)


    
    # turn the x-ticks into the positions from points
    axes['panel_C_main'].set_yticks([])
    axes['panel_C_main'].set_xticks([20, 60, 100, 140, 180], np.round(points[[20, 60, 100, 140, 180]], 2))
    axes['panel_C_main'].set_xlabel("Contrast scalar distribution", size=17)

    cbar = fig.colorbar(im1, cax=axes['panel_C_colorbar'], orientation='vertical')
    cbar.set_label('Density', rotation=270, fontsize=17, labelpad=22)
    axes['panel_C_marginal'].set_xticks([])  # Removes x ticks
    axes['panel_C_marginal'].set_yticks([])  # Removes y ticks
    axes['panel_C_main'].tick_params(axis='both', which='major', labelsize=13)

    axes['panel_C_reward'].plot(np.sort(avg_rewards), np.arange(403), c='k')
    axes['panel_C_reward'].set_ylabel("Sessions (reward rate sorted)", size=17)
    axes['panel_C_reward'].set_xlabel("Reward\nrate", size=15)
    axes['panel_C_reward'].set_xticks([0.7, 0.9], [0.7, 0.9])
    axes['panel_C_reward'].tick_params(axis='both', which='major', labelsize=13)
    axes['panel_C_reward'].set_ylim(0, 403)

    interp_scalar_means = interpolated_timeout_scalars.mean(axis=0)
    interp_scalar_stds = np.std(interpolated_timeout_scalars, axis=0)
    print(f"count of interpolated timeout scalars {interpolated_timeout_scalars.shape}")
    line1, = axes['panel_B1'].plot(np.linspace(0, 1, 100), interp_scalar_means, color='k', lw=2, label=r"$s^c$")
    axes['panel_B1'].fill_between(np.linspace(0, 1, 100), interp_scalar_means + interp_scalar_stds, interp_scalar_means - interp_scalar_stds, color='k', alpha=1/3)
    line2 = axes['panel_B1'].axhline(0.19198039866457073, color='k', ls='--', alpha=0.5, label=r"$s^c$ mode")

    ax_right = axes['panel_B1'].twinx()
    interp_reward_means = interpolated_rewards.mean(axis=0)
    interp_reward_sems = sem(interpolated_rewards, axis=0)
    print(f"count of interpolated timeout rewards {interpolated_rewards.shape}")
    line3, = ax_right.plot(np.linspace(0, 1, 100), interp_reward_means, color='g', lw=2, label="Mouse perf.")
    ax_right.errorbar(np.linspace(0, 1, 100), interp_reward_means, interp_reward_sems, color='g', alpha=1/3)
    line4 = ax_right.axhline(np.mean((input_seq[:, 1:, 5][train_mask] + 1) / 2), color='g', ls='--', alpha=0.5, label="Mean perf.")
    ax_right.set_ylim(-0.95, 1)
    ax_right.set_yticks([0, 0.5, 1], [])

    # add a fake line
    interp_reward_est_means = interpolated_reward_estimates.mean(axis=0)
    interp_reward_est_sems = sem(interpolated_reward_estimates, axis=0)
    line5 = ax_right.errorbar(np.linspace(0, 1, 100), interp_reward_est_means, interp_reward_est_sems, color='r', alpha=1/3, label="Predicted perf.")

    axes['panel_B1'].set_ylabel("Contrast scalar", fontsize=17)
    axes['panel_B1'].set_xlabel("Interpolated session time", fontsize=17)
    axes['panel_B1'].set_xlim(0, 1)
    axes['panel_B1'].set_ylim(-0.95, 1)
    axes['panel_B1'].tick_params(axis='both', which='major', labelsize=15)
    axes['panel_B1'].spines[['right', 'top']].set_visible(False)
    axes['panel_B1'].set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    axes['panel_B1'].legend(lines, labels, frameon=False, loc='lower left', fontsize=10, ncols=1)
    axes['panel_B1'].set_yticks([-0.5, 0, 0.5])


    zoomed_means = np.array([np.mean(x) for x in zoomed_timeout_scalars])
    zoomed_stds = np.array([np.std(x) for x in zoomed_timeout_scalars])
    print(f"count of zoomed timeout scalars {[len(x) for x in zoomed_timeout_scalars]}")
    axes['panel_B2'].plot(range(-before, after + 1), zoomed_means, color='k', lw=2)
    line1 = axes['panel_B2'].fill_between(range(-before, after + 1), zoomed_means - zoomed_stds, zoomed_means + zoomed_stds, color='k', alpha=1/3, label=r"$s^c$")
    line2 = axes['panel_B2'].axhline(0.19198039866457073, color='k', ls='--', alpha=0.5, label=r"$s^c$ mode")
    axes['panel_B2'].axvline(0, color='grey', alpha=0.6)

    ax_right = axes['panel_B2'].twinx()
    zoomed_reward_mean = np.array([np.mean(x) for x in zoomed_rewards])
    zoomed_reard_sems = np.array([sem(x) for x in zoomed_rewards])
    print(f"count of zoomed timeout rewards {[len(x) for x in zoomed_rewards]}")

    zoomed_rew_pred_mean = np.array([np.mean(x) for x in zoomed_reward_estimates])
    zoomed_rew_pred_sems = np.array([sem(x) for x in zoomed_reward_estimates])

    ax_right.plot(range(-before, after + 1), zoomed_reward_mean, color='g', lw=2)
    line3 = ax_right.errorbar(range(-before, after + 1), zoomed_reward_mean, zoomed_reard_sems, color='g', alpha=1/3, label="Mouse perf.")

    ax_right.plot(range(-before, after + 1), zoomed_rew_pred_mean, color='r', lw=2)
    line4 = ax_right.errorbar(range(-before, after + 1), zoomed_rew_pred_mean, zoomed_rew_pred_sems, color='r', alpha=1/3, label="Predicted perf.")

    line5 = ax_right.axhline(np.mean((input_seq[:, 1:, 5][train_mask] + 1) / 2), color='g', ls='--', alpha=0.5, label="Mean perf.")
    ax_right.set_ylabel('Reward rate', color='green', fontsize=17, rotation=270, labelpad=18)
    ax_right.set_ylim(-0.95, 1)
    ax_right.set_yticks([0, 0.5, 1])
    ax_right.tick_params(axis='both', which='major', labelsize=15)

    lines = [line3, line4, line5]
    labels = [line.get_label() for line in lines]
    axes['panel_B2'].legend(lines, labels, frameon=False, ncols=1, loc='lower left', fontsize=10)

    axes['panel_B2'].set_xlabel("Trials around timeout", fontsize=17)
    axes['panel_B2'].set_xlim(-before, after)
    axes['panel_B2'].set_ylim(-0.95, 1)
    axes['panel_B2'].tick_params(axis='both', which='major', labelsize=15)
    axes['panel_B2'].spines[['right', 'top']].set_visible(False)
    axes['panel_B2'].set_yticks([-0.5, 0, 0.5], [])

    # Panel labels
    labels = [
        {'label_text': 'a', 'xpos': 0.095, 'ypos': 0.91},
        {'label_text': 'c', 'xpos': 0.095, 'ypos': 0.455},
        {'label_text': 'd', 'xpos': 0.32, 'ypos': 0.455},
        {'label_text': 'b', 'xpos': 0.615, 'ypos': 0.91}
    ]
    for lbl in labels:
        fig.text(lbl['xpos'], lbl['ypos'], lbl['label_text'], fontsize=33, fontweight='bold')

    plt.tight_layout()
    plt.savefig("pmfs new", bbox_inches='tight', dpi=300)
    plt.show()
