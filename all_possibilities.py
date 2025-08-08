import pickle
from load_network import loaded_network
import numpy as np
import load_data
import matplotlib.pyplot as plt
import json
import jax
from scipy.stats import pearsonr
import figrid as fg
from scipy.stats import sem

model_performance_color = '#4a0100'


# always plot the two possibilities of choices from the average onwards (start next point discontinous from the next average)
# do so for one total line (compare exp filter and abcd)
# for abcd, also do so separately for the two components of the two histories

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


def plot_start_and_history(input, reload_net, ax, marker='*', second=False, **kwargs):


    if not second:
        starting_point = reload_net.final_decay_1(input[:3].reshape(1, -1), addon=input[3:].reshape(1, -1))[0]
    else:
        starting_point = reload_net.final_decay_2(input[:3].reshape(1, -1), addon=input[3:].reshape(1, -1))[0]

    starting_point = 1 / (1 + np.exp(-starting_point))

    ax.scatter(starting_point[2], starting_point[0], marker=marker, s=80, zorder=2, **kwargs)


if True:
    results = reload_net_____.lstm_fn.apply(reload_net_____.params, None, input_seq[:, :, reload_net_____.input_list])
    if not infos_____['lstm_contrast_scalar']:
        history_safe_____ = results[1]
    else:
        history_safe_____ = results[1][1]

    results_2 = reload_net_abcd.lstm_fn.apply(reload_net_abcd.params, None, input_seq[:, :, reload_net_abcd.input_list])
    if not infos_abcd['lstm_contrast_scalar']:
        history_safe_abcd = results_2[1]
    else:
        history_safe_abcd = results_2[1][1]

    history_changes = {(2, 1.): [], (0, 1.): [], (2, -1.): [], (0, -1.): []}
    history_changes_____ = {(2, 1.): [], (0, 1.): [], (2, -1.): [], (0, -1.): []}

    for sess, _ in enumerate(train_eids):

        print(sess)

        history = infos_abcd['best_net']['final_net']['history_weighting'] * history_safe_abcd[0][sess][:1+train_mask[sess].sum()] + infos_abcd['best_net']['final_net']['history_weighting_2'] * history_safe_abcd[1][sess][:1+train_mask[sess].sum()]
        history = history[:, 2] - history[:, 0]

        classic_history = infos_____['best_net']['final_net']['history_weighting'] * history_safe_____[sess][:1+train_mask[sess].sum()]
        classic_history = classic_history[:, 2] - classic_history[:, 0]

        choices = np.argmax(input_seq[sess, 1:1+train_mask[sess].sum(), :3], axis=1)
        rewards = input_seq[sess, 1:1+train_mask[sess].sum(), 5]

        for trial in range(train_mask[sess].sum()):
            if choices[trial] != 1:
                # print(trial, choices[trial], rewards[trial], history[trial], history[trial + 1])
                history_changes[(choices[trial], rewards[trial])].append((history[trial], history[trial + 1]))
                history_changes_____[(choices[trial], rewards[trial])].append((classic_history[trial], classic_history[trial + 1]))
        

    pickle.dump((history_changes, history_changes_____),
                    open("all_possibilities_changes.p", 'wb'))

if True:
    results = reload_net_____.lstm_fn.apply(reload_net_____.params, None, input_seq[:, :, reload_net_____.input_list])
    if not infos_____['lstm_contrast_scalar']:
        history_safe_____ = results[1]
    else:
        history_safe_____ = results[1][1]
    predictions_____ = reload_net_____.return_predictions(input_seq, action_subsetting=True)

    results_2 = reload_net_abcd.lstm_fn.apply(reload_net_abcd.params, None, input_seq[:, :, reload_net_abcd.input_list])
    if not infos_abcd['lstm_contrast_scalar']:
        history_safe_abcd = results_2[1]
    else:
        history_safe_abcd = results_2[1][1]
    predictions_abcd = reload_net_abcd.return_predictions(input_seq, action_subsetting=True)


    from data_playground import big_types

    biggest_type_histories_____ = {0: [], 1: [], 2: [], 3: []}
    performances_____ = {0: [], 1: [], 2: [], 3: []}

    biggest_type_histories_abcd = {0: [], 1: [], 2: [], 3: []}
    performances_abcd = {0: [], 1: [], 2: [], 3: []}

    mouse_choices = {0: [], 1: [], 2: [], 3: []}

    big_type_blocks = {0: None, 1: None, 2: None, 3: None}
    big_type_contrasts = {0: None, 1: None, 2: None, 3: None}
    big_type_corrects = {0: np.zeros(400) - 1, 1: np.zeros(400) - 1, 2: np.zeros(400) - 1, 3: np.zeros(400) - 1}

    biased_blocks = np.load("./processed_data/train_bias.npy")

    for sess, _ in enumerate(train_eids):

        print(sess)

        for i, bt in enumerate(big_types):
            if np.array_equal(np.nan_to_num(input_seq[sess, :30, 3]), bt):

                biggest_type_histories_____[i].append(infos_____['best_net']['final_net']['history_weighting'] * history_safe_____[sess])
                performances_____[i].append(predictions_____[sess])

                biggest_type_histories_abcd[i].append((infos_abcd['best_net']['final_net']['history_weighting'] * history_safe_abcd[0][sess], infos_abcd['best_net']['final_net']['history_weighting_2'] * history_safe_abcd[1][sess]))
                performances_abcd[i].append(predictions_abcd[sess])

                choices = np.argmax(input_seq[sess, 1:401, :3], axis=1)
                mouse_choices[i].append(choices)

                big_type_corrects[i][input_seq[sess, 1:401, 5] == 1] = choices[input_seq[sess, 1:401, 5] == 1]

                if big_type_blocks[i] is None:
                    big_type_blocks[i] = biased_blocks[sess]
                    big_type_contrasts[i] = input_seq[sess, :, 3:5]

    pickle.dump((biggest_type_histories_____, performances_____, biggest_type_histories_abcd, performances_abcd, mouse_choices, big_type_corrects, big_type_blocks, big_type_contrasts),
                    open("all_possibilities.p", 'wb'))



history_changes, history_changes_____ = pickle.load(open("all_possibilities_changes.p", 'rb'))

heatmap_max = 0

bins = 41
span = np.linspace(-1.05, 3.1, bins)

heatmap = np.zeros((bins-1, bins-1))
for key in history_changes_____:
    changes = np.array(history_changes_____[key])
    for bin in range(bins - 1):
        heatmap[bin] = np.histogram(changes[np.logical_and(span[bin] < changes[:, 0], changes[:, 0] < span[bin + 1]), 1], span)[0]
    heatmap_max = max(heatmap_max, heatmap.max())


if False:
    bins = 41
    span = np.linspace(-1.05, 3.1, bins)

    heatmap = np.zeros((bins-1, bins-1))

    for key in history_changes:
        print(key)
        changes = np.array(history_changes[key])

        print(changes[changes[:, 0] > 0, 1].shape)
        print(changes[changes[:, 0] < 0, 1].shape)
        print(changes[changes[:, 0] > 0, 1].max(), changes[changes[:, 0] > 0, 1].min())

        for bin in range(bins - 1):
            # if span[bin + 1] < 0:
            #     continue
            heatmap[bin] = np.histogram(changes[np.logical_and(span[bin] < changes[:, 0], changes[:, 0] < span[bin + 1]), 1], span)[0]

        plt.imshow(heatmap.T, origin='lower')
        plt.plot([0, 39], [0, 39], 'k', lw=3)

        plt.xticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])
        plt.yticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])

        plt.ylabel("Ends up at")
        plt.xlabel("Starting value")
        plt.title(str(key) + " ABCD")

        plt.tight_layout()
        plt.savefig(str(key).replace('.', '_') + "_ABCD")
        plt.close()


        changes = np.array(history_changes_____[key])

        for bin in range(bins - 1):
            # if span[bin + 1] < 0:
            #     continue
            heatmap[bin] = np.histogram(changes[np.logical_and(span[bin] < changes[:, 0], changes[:, 0] < span[bin + 1]), 1], span)[0]

        plt.imshow(heatmap.T, origin='lower')
        plt.plot([0, 39], [0, 39], 'k', lw=3)

        plt.xticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])
        plt.yticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])

        plt.ylabel("Ends up at")
        plt.xlabel("Starting value")
        plt.title(str(key) + " abcd")

        plt.tight_layout()
        plt.savefig(str(key).replace('.', '_') + "_abcd")
        plt.close()



left_act_rewarded = np.array([0, 0, 1, 1])
left_act_unrewarded = np.array([0, 0, 1, -1])
right_act_rewarded = np.array([1, 0, 0, 1])
right_act_unrewarded = np.array([1, 0, 0, -1])


biggest_type_histories_____, performances_____, biggest_type_histories_abcd, performances_abcd, mouse_choices, big_type_corrects, big_type_blocks, big_type_contrasts = pickle.load(open("all_possibilities.p", 'rb'))

if False:
    plt.figure(figsize=(12, 7))
    history_simple = np.concatenate([x[:400] for x in biggest_type_histories_____[0]])
    history_concat = np.concatenate([x[0][:400] + x[1][:400] for x in biggest_type_histories_abcd[0]])

    ax1 = plt.subplot(2, 1, 1)
    plt.scatter(history_simple[:, 0], history_simple[:, 2], alpha=0.01)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel("Left history logit")
    ax1.set_ylabel("Right history logit")

    ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    plt.scatter(history_concat[:, 0], history_concat[:, 2], alpha=0.01)

    xmin, xmax = ax1.set_xlim(-1, 5)
    ax2.set_aspect('equal', 'box')

    ax1.plot([-1, 5], [-1, 5], 'k', alpha=0.1)
    ax2.plot([-1, 5], [-1, 5], 'k', alpha=0.1)
    ax2.set_xlabel("Left history logit")
    ax2.set_ylabel("Right history logit")


    ax1.set_xlim(-1, 5)
    ax2.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    ax2.set_ylim(-1, 5)

    plt.tight_layout()
    plt.savefig("history terms")
    plt.show()


from_trial, to_trial = 0, 300
label_size = 18

if True:
    for bt in range(4):
        fig = plt.figure(figsize=(16, 10))
        ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.05, 0.31], yspan=[0., 0.48]),
            'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.05, 0.31], yspan=[0.52, 1.]),
            'panel_B1': fg.place_axes_on_grid(fig, xspan=[0.39, .57], yspan=[0., 0.18]),
            'panel_B2': fg.place_axes_on_grid(fig, xspan=[0.39, .57], yspan=[0.19, 0.37]),
            'panel_B3': fg.place_axes_on_grid(fig, xspan=[0.58, .76], yspan=[0., 0.18]),
            'panel_B4': fg.place_axes_on_grid(fig, xspan=[0.58, .76], yspan=[0.19, 0.37]),
            'panel_B5': fg.place_axes_on_grid(fig, xspan=[0.77, .95], yspan=[0., 0.18]),
            'panel_B6': fg.place_axes_on_grid(fig, xspan=[0.77, .95], yspan=[0.19, 0.37]),
            'panel_B_colorbar': fg.place_axes_on_grid(fig, xspan=[0.98, 1.], yspan=[0., 0.37]),
            'panel_C': fg.place_axes_on_grid(fig, xspan=[0.39, 1.], yspan=[0.46, 0.64]),
            'panel_D': fg.place_axes_on_grid(fig, xspan=[0.39, 1.], yspan=[0.66, 0.89]),
            'panel_B_sub': fg.place_axes_on_grid(fig, xspan=[0.39,  1.], yspan=[0.91, 1.])}

        heatmap = np.zeros((bins-1, bins-1))
        for key, axes, title in zip([(2, 1), (0, 1)], [ax['panel_B1'], ax['panel_B2']], ["abcd", ""]):
            changes = np.array(history_changes_____[key])

            for bin in range(bins - 1):
                # if span[bin + 1] < 0:
                #     continue
                heatmap[bin] = np.histogram(changes[np.logical_and(span[bin] < changes[:, 0], changes[:, 0] < span[bin + 1]), 1], span)[0]

            im1 = axes.imshow(heatmap.T, origin='lower', aspect='auto')
            axes.plot([0, 39], [0, 39], 'k', lw=2)

            axes.set_xticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])
            axes.set_yticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])
            axes.set_title(title, fontsize=16)

            if key == (2, 1):
                axes.set_xticks([])
            else:
                axes.set_xticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])

        cbar = fig.colorbar(im1, cax=ax['panel_B_colorbar'], orientation='vertical', ticks=[0, 3337.0])
        cbar.set_label('# of occurences', rotation=270, fontsize=17, labelpad=-5)
        cbar.ax.set_yticklabels(['Low', 'High'], fontsize=13)


        for key, axes, title in zip([(2, 1), (0, 1), (2, -1), (0, -1)], [ax['panel_B3'], ax['panel_B4'], ax['panel_B5'], ax['panel_B6']],
                                    ["Rew., ABCD", "", "Unrew., ABCD", ""]):
            changes = np.array(history_changes[key])

            for bin in range(bins - 1):
                # if span[bin + 1] < 0:
                #     continue
                heatmap[bin] = np.histogram(changes[np.logical_and(span[bin] < changes[:, 0], changes[:, 0] < span[bin + 1]), 1], span)[0]

            axes.imshow(heatmap.T, origin='lower', aspect='auto')
            axes.plot([0, 39], [0, 39], 'k', lw=2)

            axes.set_yticks([])
            axes.set_title(title, fontsize=16)

            if key in [(2, 1), (2, -1)]:
                axes.set_xticks([])
            else:
                axes.set_xticks([0, 10, 20, 30], [np.round(span[x], 2) for x in [0, 10, 20, 30]])

        fig.text(0.38, 0.6, r"Updated $\Delta$ hist. logit", fontsize=label_size-1, rotation='vertical')
        fig.text(0.6, 0.55, r"Starting $\Delta$ hist. logit", fontsize=label_size-1)
        fig.text(0.865, 0.75, r"Rightwards", fontsize=label_size-3, rotation=-90)
        fig.text(0.865, 0.625, r"Leftwards", fontsize=label_size-3, rotation=-90)


        for i, local_ax in enumerate([ax['panel_A_1'], ax['panel_A_2']]):
            x = plot_start_and_history(left_act_rewarded, reload_net_abcd, local_ax, c='b', label="Left choice, rewarded", second=i == 0)
            x = plot_start_and_history(right_act_rewarded, reload_net_abcd, local_ax, c='r', label="Right choice, rewarded", second=i == 0)
            x = plot_start_and_history(left_act_unrewarded, reload_net_abcd, local_ax, c='b', marker='d', label="Left choice, unrewarded", edgecolors='b', lw=2, second=i == 0)
            x = plot_start_and_history(right_act_unrewarded, reload_net_abcd, local_ax, c='r', marker='d', label="Right choice, unrewarded", edgecolors='r', lw=2, second=i == 0)
            local_ax.plot([0, 1], [0, 1], 'k', alpha=1/2)
            normal_decay = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
            local_ax.scatter(normal_decay, normal_decay, marker='o', s=80, c='k', alpha=0.5, zorder=2, label="abcd decay")
            if i == 1:
                local_ax.set_xlim(0.95, 1.002)
                local_ax.set_ylim(0.95, 1.002)
                local_ax.axhline(1, color='grey', alpha=0.7, zorder=-10)
                local_ax.axvline(1, color='grey', alpha=0.7, zorder=-10)
                local_ax.set_xlabel("Leftwards logit decay", fontsize=label_size)
                local_ax.set_ylabel("Rightwards logit decay", fontsize=label_size)
                local_ax.set_aspect('equal', adjustable='box')
            else:
                local_ax.set_xlim(0., 1)
                local_ax.set_ylim(0., 1)

                local_ax.set_aspect('equal', adjustable='box')
                local_ax.legend(frameon=False)
        ax['panel_A_1'].set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax['panel_A_1'].set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax['panel_A_1'].set_title("Fast decay", fontsize=label_size)
        ax['panel_A_2'].set_title("Slow decay", fontsize=label_size)
        ax['panel_A_2'].set_xticks([0.95, 0.975, 1])
        ax['panel_A_2'].set_yticks([0.95, 0.975, 1])



        # hist_1 = np.array([x for x in biggest_type_histories_____[bt]]).mean(axis=0)
        # ax['panel_B'].plot(hist_1[:, 2] - hist_1[:, 0], label='---- history')
        hist_2_big = np.array([x[0] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
        hist_2_small = np.array([x[1] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
        hist_2 = hist_2_big + hist_2_small
        # ax['panel_B'].plot(hist_2[:, 2] - hist_2[:, 0], label='ABCD history')
        # ax['panel_B'].axhline(0, color='grey', alpha=0.3)

        # ax['panel_B'].scatter(np.arange(0, 400), big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1], color='k', alpha=0.3)
        zero_contrasts = (big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1]) == 0
        # ax['panel_B'].scatter(np.arange(0, 400)[zero_contrasts], np.zeros(400)[zero_contrasts], edgecolors='k', color='white', alpha=0.6, s=20)

        ax['panel_C'].plot(-hist_2_big[:, 0], 'blue', label='leftwards logit')
        ax['panel_C'].plot(hist_2_big[:, 2], 'red', label='rightwards logit')
        ax['panel_C'].plot(hist_2_big[:, 2] - hist_2_big[:, 0], 'k', alpha=0.2, label='logit difference')
        ax['panel_C'].axhline(0, color='grey', alpha=0.3)


        ax['panel_D'].plot(-hist_2_small[:, 0], 'blue')
        ax['panel_D'].plot(hist_2_small[:, 2], 'red')
        ax['panel_D'].plot(hist_2_small[:, 2] - hist_2_small[:, 0], 'k', alpha=0.2)
        ax['panel_D'].axhline(0, color='grey', alpha=0.3)

        red_choice_movement = np.zeros(400)
        blue_choice_movement = np.zeros(400)

        for trial in range(0, 400):

            # simulate a rightwards choice:

            counter_fact_choice = np.array([1, 0, 0])
            reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
            local_decay = reload_net_abcd.final_decay_1(counter_fact_choice.reshape(1, -1), addon=reward)

            # if 280 < trial < 290:
            #     print(trial, counter_fact_choice, reward, jax.nn.sigmoid(local_decay[0]), hist_2_big[trial])

            ax['panel_C'].plot([trial, trial+1], [-hist_2_big[trial, 0], -hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - infos_abcd['best_net']['final_net']['history_weighting'][0]], 'b', ls='--', alpha=0.2)
            ax['panel_C'].plot([trial, trial+1], [hist_2_big[trial, 2], hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2])], 'b', ls='--', alpha=0.2)
            blue_choice_movement[trial] += hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting'][0] - hist_2_big[trial, 0] - (hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) - hist_2_big[trial, 2])

            counter_fact_choice = np.array([0, 0, 1])
            reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 2 else -np.ones((1, 1))
            local_decay = reload_net_abcd.final_decay_1(counter_fact_choice.reshape(1, -1), addon=reward)

            # if 280 < trial < 290:
            #     print(trial, counter_fact_choice, reward, jax.nn.sigmoid(local_decay[0]), hist_2_big[trial])
            #     print()

            ax['panel_C'].plot([trial, trial+1], [-hist_2_big[trial, 0], -hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0])], 'r', ls='--', alpha=0.2)
            ax['panel_C'].plot([trial, trial+1], [hist_2_big[trial, 2], hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting'][0]], 'r', ls='--', alpha=0.2)
            red_choice_movement[trial] += hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting'][0] - hist_2_big[trial, 2] - (hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - hist_2_big[trial, 0])


            counter_fact_choice = np.array([1, 0, 0])
            reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
            local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

            ax['panel_D'].plot([trial, trial+1], [-hist_2_small[trial, 0], -hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'b', ls='--', alpha=0.2)
            ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2])], 'b', ls='--', alpha=0.2, label="Change from leftwards choice" if trial == 0 else None)
            blue_choice_movement[trial] += hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 0] - (hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) - hist_2_small[trial, 2])

            counter_fact_choice = np.array([0, 0, 1])
            reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 2 else -np.ones((1, 1))
            local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

            ax['panel_D'].plot([trial, trial+1], [-hist_2_small[trial, 0], -hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0])], 'r', ls='--', alpha=0.2, label="Change from rightwards choice" if trial == 0 else None)
            ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'r', ls='--', alpha=0.2)
            red_choice_movement[trial] += hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 2] - (hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - hist_2_small[trial, 0])

        # diff = np.array([x[:400] for x in performances_abcd[bt]]).mean(axis=0) - np.array([x[:400] for x in performances_____[bt]]).mean(axis=0)
        all_diffs = [np.array(x[:400]) - np.array(y[:400]) for x, y, in zip(performances_abcd[bt], performances_____[bt])]
        diff = np.mean(all_diffs, 0)
        print(f"number of sessions in type {bt}: {len(all_diffs)}")

        all_windowed_diffs = [[np.sum(x[i-10:i]) for i in range(10, len(x))] for x in all_diffs]
        windowed_diff = np.mean(all_windowed_diffs, 0)
        windowed_diff_sems = sem(all_windowed_diffs, 0)
        # ax['panel_B_sub'].plot(range(10, len(diff)), windowed_diff, c='green', label="10 Trial model perf. diff.", alpha=0.7)
        # ax['panel_B_sub'].fill_between(range(10, len(diff)), windowed_diff + windowed_diff_sems, windowed_diff - windowed_diff_sems, color='green', alpha=0.2)
        ax['panel_B_sub'].errorbar(range(10, len(diff)), windowed_diff, windowed_diff_sems, c=model_performance_color, label="10 Trial model perf. diff.", alpha=0.7)
        ax['panel_B_sub'].legend(frameon=False)

        normal_decay = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
        # ax['panel_B'].fill_between(np.arange(1, 401), (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay - reload_net_____.infos['best_net']['final_net']['history_weighting'][0],
        #                            (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay + reload_net_____.infos['best_net']['final_net']['history_weighting'][0], color='purple', alpha=0.3, label="---- 1 choice range")
        # ax['panel_B'].fill_between(np.arange(1, 401), (hist_2[:, 2] - hist_2[:, 0])[0:400] + red_choice_movement, (hist_2[:, 2] - hist_2[:, 0])[0:400] - blue_choice_movement, color='green', alpha=0.3, label="ABCD 1 choice range")
        # ax['panel_B'].legend(frameon=False, ncols=2)

        for local_ax in [ax['panel_B_sub'], ax['panel_C'], ax['panel_D']]:
            previous = 0.5
            start = 0

            local_ax.set_xlim(from_trial, to_trial)
            local_ax.set_xlim(from_trial, to_trial)
            local_ax.set_xlim(from_trial, to_trial)

            for i in range(400):
                if big_type_blocks[bt][i] == 0.2 and previous != 0.2:
                    if previous == 0.5:
                        start = i
                        previous = 0.2
                    else:
                        local_ax.axvspan(start, i, color='r', alpha=0.1)
                        start = i
                        previous = 0.2
                elif big_type_blocks[bt][i] == 0.8 and previous != 0.8:
                    if previous == 0.5:
                        start = i
                        previous = 0.8
                    else:
                        local_ax.axvspan(start, i, color='b', alpha=0.1)
                        start = i
                        previous = 0.8
                if big_type_blocks[bt][i] == 0. or i == 399:
                    if previous == 0.2:
                        local_ax.axvspan(start, i, color='b', alpha=0.1)
                    elif previous == 0.8:
                        local_ax.axvspan(start, i, color='r', alpha=0.1)
                    break

        # ax['panel_B'].set_xticks([])
        ax['panel_C'].set_xticks([])
        ax['panel_B_sub'].axhline(0, color='k', alpha=0.2)
        ax['panel_D'].set_xticks([])
        ax['panel_D'].legend(frameon=False)
        ax['panel_B_sub'].set_xlabel("Trial", fontsize=label_size)
        ax['panel_C'].legend(frameon=False)

        # fig.text(0.375, 0.38, "Logits", fontsize=label_size, ha='left', va='top', rotation='vertical')
        # fig.text(0.38, 0.65, r"$\Delta$perf.", fontsize=label_size-2, ha='left', va='top', rotation='vertical')
        # fig.text(0.375, 0.85, "Logit diff.", fontsize=label_size, ha='left', va='top', rotation='vertical')
        # ax['panel_B'].set_ylabel(r"$\Delta$ logits", fontsize=label_size)
        ax['panel_B_sub'].set_ylabel(r"$\Delta$ perf.", fontsize=label_size-2)
        ax['panel_D'].set_ylabel("Fast logits", fontsize=label_size)
        ax['panel_C'].set_ylabel("Slow logits", fontsize=label_size)

        # Panel labels
        labels = [
            {'label_text': 'a', 'xpos': 0.11, 'ypos': 0.88},
            {'label_text': 'b', 'xpos': 0.11, 'ypos': 0.47},
            {'label_text': 'c', 'xpos': 0.375, 'ypos': 0.88},
            {'label_text': 'd', 'xpos': 0.375, 'ypos': 0.535},
            {'label_text': 'e', 'xpos': 0.375, 'ypos': 0.355},
            {'label_text': 'f', 'xpos': 0.375, 'ypos': 0.183}
        ]
        for lbl in labels:
            fig.text(lbl['xpos'], lbl['ypos'], lbl['label_text'], fontsize=33, fontweight='bold')

        plt.savefig(f"figures/all_possibilities_{bt}.png", bbox_inches='tight', dpi=300)
        plt.show()

        break

### Plot supplementary figures

from_trial, to_trial = 0, 400

for bt in range(4):
    fig = plt.figure(figsize=(16, 10))
    ax = {'panel_C': fg.place_axes_on_grid(fig, xspan=[0., 1.], yspan=[0., 0.37]),
          'panel_D': fg.place_axes_on_grid(fig, xspan=[0., 1.], yspan=[0.4, 0.77]),
          'panel_B_sub': fg.place_axes_on_grid(fig, xspan=[0.,  1.], yspan=[0.82, 1.])}


    # hist_1 = np.array([x for x in biggest_type_histories_____[bt]]).mean(axis=0)
    # ax['panel_B'].plot(hist_1[:, 2] - hist_1[:, 0], label='---- history')
    hist_2_big = np.array([x[0] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
    hist_2_small = np.array([x[1] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
    hist_2 = hist_2_big + hist_2_small
    # ax['panel_B'].plot(hist_2[:, 2] - hist_2[:, 0], label='ABCD history')
    # ax['panel_B'].axhline(0, color='grey', alpha=0.3)

    # ax['panel_B'].scatter(np.arange(0, 400), big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1], color='k', alpha=0.3)
    zero_contrasts = (big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1]) == 0
    # ax['panel_B'].scatter(np.arange(0, 400)[zero_contrasts], np.zeros(400)[zero_contrasts], edgecolors='k', color='white', alpha=0.6, s=20)

    ax['panel_C'].plot(-hist_2_big[:, 0], 'blue', label='leftwards logit')
    ax['panel_C'].plot(hist_2_big[:, 2], 'red', label='rightwards logit')
    ax['panel_C'].plot(hist_2_big[:, 2] - hist_2_big[:, 0], 'k', alpha=0.2, label='logit difference')
    ax['panel_C'].axhline(0, color='grey', alpha=0.3)


    ax['panel_D'].plot(-hist_2_small[:, 0], 'blue')
    ax['panel_D'].plot(hist_2_small[:, 2], 'red')
    ax['panel_D'].plot(hist_2_small[:, 2] - hist_2_small[:, 0], 'k', alpha=0.2)
    ax['panel_D'].axhline(0, color='grey', alpha=0.3)

    red_choice_movement = np.zeros(400)
    blue_choice_movement = np.zeros(400)

    for trial in range(0, 400):

        # simulate a rightwards choice:

        counter_fact_choice = np.array([1, 0, 0])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_1(counter_fact_choice.reshape(1, -1), addon=reward)

        # if 280 < trial < 290:
        #     print(trial, counter_fact_choice, reward, jax.nn.sigmoid(local_decay[0]), hist_2_big[trial])

        ax['panel_C'].plot([trial, trial+1], [-hist_2_big[trial, 0], -hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - infos_abcd['best_net']['final_net']['history_weighting'][0]], 'b', ls='--', alpha=0.2)
        ax['panel_C'].plot([trial, trial+1], [hist_2_big[trial, 2], hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2])], 'b', ls='--', alpha=0.2)
        blue_choice_movement[trial] += hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting'][0] - hist_2_big[trial, 0] - (hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) - hist_2_big[trial, 2])

        counter_fact_choice = np.array([0, 0, 1])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 2 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_1(counter_fact_choice.reshape(1, -1), addon=reward)

        # if 280 < trial < 290:
        #     print(trial, counter_fact_choice, reward, jax.nn.sigmoid(local_decay[0]), hist_2_big[trial])
        #     print()

        ax['panel_C'].plot([trial, trial+1], [-hist_2_big[trial, 0], -hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0])], 'r', ls='--', alpha=0.2)
        ax['panel_C'].plot([trial, trial+1], [hist_2_big[trial, 2], hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting'][0]], 'r', ls='--', alpha=0.2)
        red_choice_movement[trial] += hist_2_big[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting'][0] - hist_2_big[trial, 2] - (hist_2_big[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - hist_2_big[trial, 0])


        counter_fact_choice = np.array([1, 0, 0])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

        ax['panel_D'].plot([trial, trial+1], [-hist_2_small[trial, 0], -hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'b', ls='--', alpha=0.2)
        ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2])], 'b', ls='--', alpha=0.2, label="Change from leftwards choice" if trial == 0 else None)
        blue_choice_movement[trial] += hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 0] - (hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) - hist_2_small[trial, 2])

        counter_fact_choice = np.array([0, 0, 1])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 2 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

        ax['panel_D'].plot([trial, trial+1], [-hist_2_small[trial, 0], -hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0])], 'r', ls='--', alpha=0.2, label="Change from rightwards choice" if trial == 0 else None)
        ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'r', ls='--', alpha=0.2)
        red_choice_movement[trial] += hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 2] - (hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - hist_2_small[trial, 0])

    # diff = np.array([x[:400] for x in performances_abcd[bt]]).mean(axis=0) - np.array([x[:400] for x in performances_____[bt]]).mean(axis=0)
    all_diffs = [np.array(x[:400]) - np.array(y[:400]) for x, y, in zip(performances_abcd[bt], performances_____[bt])]
    print(f"number of sessions in type {bt}: {len(all_diffs)}")
    diff = np.mean(all_diffs, 0)

    all_windowed_diffs = [[np.sum(x[i-10:i]) for i in range(10, len(x))] for x in all_diffs]
    windowed_diff = np.mean(all_windowed_diffs, 0)
    windowed_diff_sems = sem(all_windowed_diffs, 0)
    # ax['panel_B_sub'].plot(range(10, len(diff)), windowed_diff, c='green', label="10 Trial model perf. diff.", alpha=0.7)
    # ax['panel_B_sub'].fill_between(range(10, len(diff)), windowed_diff + windowed_diff_sems, windowed_diff - windowed_diff_sems, color='green', alpha=0.2)
    ax['panel_B_sub'].errorbar(range(10, len(diff)), windowed_diff, windowed_diff_sems, c=model_performance_color, label="10 Trial model perf. diff.", alpha=0.7)
    ax['panel_B_sub'].legend(frameon=False)

    normal_decay = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
    # ax['panel_B'].fill_between(np.arange(1, 401), (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay - reload_net_____.infos['best_net']['final_net']['history_weighting'][0],
    #                            (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay + reload_net_____.infos['best_net']['final_net']['history_weighting'][0], color='purple', alpha=0.3, label="---- 1 choice range")
    # ax['panel_B'].fill_between(np.arange(1, 401), (hist_2[:, 2] - hist_2[:, 0])[0:400] + red_choice_movement, (hist_2[:, 2] - hist_2[:, 0])[0:400] - blue_choice_movement, color='green', alpha=0.3, label="ABCD 1 choice range")
    # ax['panel_B'].legend(frameon=False, ncols=2)

    for local_ax in [ax['panel_B_sub'], ax['panel_C'], ax['panel_D']]:
        previous = 0.5
        start = 0

        local_ax.set_xlim(from_trial, to_trial)
        local_ax.set_xlim(from_trial, to_trial)
        local_ax.set_xlim(from_trial, to_trial)

        for i in range(400):
            if big_type_blocks[bt][i] == 0.2 and previous != 0.2:
                if previous == 0.5:
                    start = i
                    previous = 0.2
                else:
                    local_ax.axvspan(start, i, color='r', alpha=0.1)
                    start = i
                    previous = 0.2
            elif big_type_blocks[bt][i] == 0.8 and previous != 0.8:
                if previous == 0.5:
                    start = i
                    previous = 0.8
                else:
                    local_ax.axvspan(start, i, color='b', alpha=0.1)
                    start = i
                    previous = 0.8
            if big_type_blocks[bt][i] == 0. or i == 399:
                if previous == 0.2:
                    local_ax.axvspan(start, i, color='b', alpha=0.1)
                elif previous == 0.8:
                    local_ax.axvspan(start, i, color='r', alpha=0.1)
                break

    # ax['panel_B'].set_xticks([])
    ax['panel_C'].set_xticks([])
    ax['panel_B_sub'].axhline(0, color='k', alpha=0.2)
    ax['panel_D'].set_xticks([])
    ax['panel_D'].legend(frameon=False)
    ax['panel_B_sub'].set_xlabel("Trial", fontsize=label_size)
    ax['panel_C'].legend(frameon=False)

    # fig.text(0.375, 0.38, "Logits", fontsize=label_size, ha='left', va='top', rotation='vertical')
    # fig.text(0.38, 0.65, r"$\Delta$perf.", fontsize=label_size-2, ha='left', va='top', rotation='vertical')
    # fig.text(0.375, 0.85, "Logit diff.", fontsize=label_size, ha='left', va='top', rotation='vertical')
    # ax['panel_B'].set_ylabel(r"$\Delta$ logits", fontsize=label_size)
    ax['panel_B_sub'].set_ylabel(r"$\Delta$ perf.", fontsize=label_size-2)
    ax['panel_D'].set_ylabel("Fast logits", fontsize=label_size)
    ax['panel_C'].set_ylabel("Slow logits", fontsize=label_size)

    # Panel labels
    labels = [
        {'label_text': 'a', 'xpos': 0.08, 'ypos': 0.88},
        {'label_text': 'b', 'xpos': 0.08, 'ypos': 0.58},
        {'label_text': 'c', 'xpos': 0.08, 'ypos': 0.25}
    ]
    for lbl in labels:
        fig.text(lbl['xpos'], lbl['ypos'], lbl['label_text'], fontsize=33, fontweight='bold')

    plt.savefig(f"figures/all_possibilities_no_redundancy_{bt}.png", bbox_inches='tight', dpi=300)
    plt.close()