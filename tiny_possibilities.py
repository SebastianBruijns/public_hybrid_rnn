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

mouse_performance_color = 'green'
model_performance_color = '#4a0100' # mahogany

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


# best abcd
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

if False:
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
                    open("temp_saves/all_possibilities.p", 'wb'))
    quit()

left_act_rewarded = np.array([0, 0, 1, 1])
left_act_unrewarded = np.array([0, 0, 1, -1])
right_act_rewarded = np.array([1, 0, 0, 1])
right_act_unrewarded = np.array([1, 0, 0, -1])


biggest_type_histories_____, performances_____, biggest_type_histories_abcd, performances_abcd, mouse_choices, big_type_corrects, big_type_blocks, big_type_contrasts = pickle.load(open("temp_saves/all_possibilities.p", 'rb'))


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

cont_map = {0: 0, 0.0625: 0.25, 0.125: 0.5, 0.25: 0.75, 1: 1}

for bt, (from_trial, to_trial) in enumerate(zip([323, 185, 205, 210], [338, 200, 220, 225])):
    fig = plt.figure(figsize=(16, 8))
    ax = {'panel_B': fg.place_axes_on_grid(fig, xspan=[0., 1.], yspan=[0., 0.37]),
          'panel_B_sub': fg.place_axes_on_grid(fig, xspan=[0., 1.], yspan=[0.38, 0.51]),
          'panel_C': fg.place_axes_on_grid(fig, xspan=[0., 1.], yspan=[0.53, 0.76]),
          'panel_D': fg.place_axes_on_grid(fig, xspan=[0.,  1.], yspan=[0.77, 1.])}


    hist_1 = np.array([x for x in biggest_type_histories_____[bt]]).mean(axis=0)
    all_of_them = np.array([x for x in biggest_type_histories_____[bt]])
    hist_1_sem = sem(all_of_them[..., 2] - all_of_them[..., 0])
    hist_2_big = np.array([x[0] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
    hist_2_small = np.array([x[1] for x in biggest_type_histories_abcd[bt]]).mean(axis=0)
    hist_2 = hist_2_big + hist_2_small
    a, b = np.array([x[0] for x in biggest_type_histories_abcd[bt]]), np.array([x[1] for x in biggest_type_histories_abcd[bt]])
    hist_2_sem = sem(a[..., 2] - a[..., 0] + (b[..., 2] - b[..., 0]), 0)
    ax['panel_B'].axhline(0, color='grey', alpha=0.3)

    # ax['panel_B'].plot(hist_2[:, 2] - hist_2[:, 0], label='ABCD history', c='orange')
    # ax['panel_B'].fill_between(np.arange(1526), hist_2[:, 2] - hist_2[:, 0] + hist_2_sem, hist_2[:, 2] - hist_2[:, 0] - hist_2_sem, color='orange', alpha=0.2)
    ax['panel_B'].errorbar(np.arange(1526), hist_2[:, 2] - hist_2[:, 0], hist_2_sem, label='ABCD history', c='orange', alpha=0.8)
    
    # ax['panel_B'].plot(hist_1[:, 2] - hist_1[:, 0], label='abcd history', c='m')
    # ax['panel_B'].fill_between(np.arange(1526), hist_1[:, 2] - hist_1[:, 0] + hist_1_sem, hist_1[:, 2] - hist_1[:, 0] - hist_1_sem, color='m', alpha=0.2)
    ax['panel_B'].errorbar(np.arange(1526), hist_1[:, 2] - hist_1[:, 0], hist_1_sem, label='abcd history', c='m', alpha=0.8)

    # ax['panel_B'].scatter(np.arange(0, 400), big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1], color='k', alpha=0.3)
    zero_contrasts = (big_type_contrasts[bt][0:400, 0] - big_type_contrasts[bt][0:400, 1]) == 0
    # ax['panel_B'].scatter(np.arange(0, 400)[zero_contrasts], np.zeros(400)[zero_contrasts], edgecolors='k', color='white', alpha=0.6, s=20)

    ax['panel_C'].plot(hist_2_small[:, 0], 'blue', label="Fast history leftwards logit")
    # ax['panel_C'].axhline(0, color='grey', alpha=0.3)


    ax['panel_D'].plot(hist_2_small[:, 2], 'red', label="Fast history rightwards logit")
    red_choice_movement = np.zeros(400)
    blue_choice_movement = np.zeros(400)

    for trial in range(0, 400):

        # simulate a rightwards choice:

        counter_fact_choice = np.array([1, 0, 0])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_1(counter_fact_choice.reshape(1, -1), addon=reward)

        counter_fact_choice = np.array([1, 0, 0])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 0 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

        ax['panel_C'].plot([trial, trial+1], [hist_2_small[trial, 0], hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'b', ls='--', alpha=0.2, label="Change from leftwards choice" if trial == 0 else None)
        ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2])], 'b', ls='--', alpha=0.2)
        blue_choice_movement[trial] += hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 0] - (hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) - hist_2_small[trial, 2])

        counter_fact_choice = np.array([0, 0, 1])
        reward = np.ones((1, 1)) if big_type_corrects[bt][trial] == 2 else -np.ones((1, 1))
        local_decay = reload_net_abcd.final_decay_2(counter_fact_choice.reshape(1, -1), addon=reward)

        ax['panel_C'].plot([trial, trial+1], [hist_2_small[trial, 0], hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0])], 'r', ls='--', alpha=0.2)
        ax['panel_D'].plot([trial, trial+1], [hist_2_small[trial, 2], hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0]], 'r', ls='--', alpha=0.2, label="Change from rightwards choice" if trial == 0 else None)
        red_choice_movement[trial] += hist_2_small[trial, 2] * jax.nn.sigmoid(local_decay[0, 2]) + infos_abcd['best_net']['final_net']['history_weighting_2'][0] - hist_2_small[trial, 2] - (hist_2_small[trial, 0] * jax.nn.sigmoid(local_decay[0, 0]) - hist_2_small[trial, 0])


    all_diffs = [np.array(x[:400]) - np.array(y[:400]) for x, y, in zip(performances_abcd[bt], performances_____[bt])]
    diff = np.mean(all_diffs, 0)

    diff_sems = sem(all_diffs, 0)
    # ax['panel_B_sub'].plot(diff, c='green', label="Trialwise model perf. diff.", alpha=0.7)
    # ax['panel_B_sub'].fill_between(range(len(diff)), diff + diff_sems, diff - diff_sems, color='green', alpha=0.2)
    ax['panel_B_sub'].errorbar(range(len(diff)), diff, diff_sems, c=model_performance_color, label="Trialwise model perf. diff.", alpha=0.7)

    ax_right = ax['panel_B_sub'].twinx()

    mouse_performance = (mouse_choices[bt] == big_type_corrects[bt]).mean(0)
    # ax['panel_B_sub'].plot(mouse_performance, c='orange', label="Mouse performance", alpha=0.7)
    # Plot on the right y-axis
    # ax_right.plot(mouse_performance, color='red', label="Mouse perf.")
    # ax_right.fill_between(range(len(mouse_performance)), mouse_performance + sem(mouse_choices[bt] == big_type_corrects[bt], 0),
    #                                                      mouse_performance - sem(mouse_choices[bt] == big_type_corrects[bt], 0), color='red', alpha=0.2)
    ax_right.errorbar(range(len(mouse_performance)), mouse_performance, sem(mouse_choices[bt] == big_type_corrects[bt], 0), color=mouse_performance_color, label="Mouse perf.", alpha=0.7)

    # Optionally, set label and other axis properties
    ax_right.set_ylabel('Mouse perf.', color=mouse_performance_color, fontsize=label_size-4, rotation=270, labelpad=13)
    ax_right.tick_params(axis='y')

    ax['panel_B_sub'].legend(frameon=False, loc='lower center', borderpad=-0.25)
    ax_right.legend(frameon=False, loc='lower right', borderpad=-0.25)

    # align y=0 across axes (thank GPT)
    ax_left = ax['panel_B_sub']

    ax_left.set_ylim(-0.04, None)
    left_bottom, left_top = ax_left.get_ylim()

    # Desired top of right axis
    right_top = 1

    # Compute relative position of y=0 on left axis
    left_zero_frac = (0 - left_bottom) / (left_top - left_bottom)

    # Compute the right axis bottom so that 0 aligns vertically
    right_bottom = (0 - right_top * left_zero_frac) / (1 - left_zero_frac)

    # Set right axis limits
    ax_right.set_ylim(right_bottom, right_top)


    normal_decay = jax.nn.sigmoid(reload_net_____.final_decay_1(np.array(0).reshape(1, -1)))[0, 0]
    # ax['panel_B'].fill_between(np.arange(1, 401), (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay - reload_net_____.infos['best_net']['final_net']['history_weighting'][0],
    #                            (hist_1[:, 2] - hist_1[:, 0])[0:400] * normal_decay + reload_net_____.infos['best_net']['final_net']['history_weighting'][0], color='purple', alpha=0.3, label="abcd 1 choice range")#, hatch='\\')
    # ax['panel_B'].fill_between(np.arange(1, 401), (hist_2[:, 2] - hist_2[:, 0])[0:400] + red_choice_movement, (hist_2[:, 2] - hist_2[:, 0])[0:400] - blue_choice_movement, color='green', alpha=0.3, label="ABCD 1 choice range")

    # plot contrasts
    ax['panel_B'].scatter(np.arange(0, 400), (big_type_corrects[bt] - 1) * 1.8, c=[-cont_map[x] for x in (big_type_contrasts[bt][0:400, 1] + big_type_contrasts[bt][0:400, 0])], cmap='gray', edgecolors='black', alpha=0.3, label="Contrasts")

    ax['panel_B'].legend(frameon=False, ncols=2)

    for local_ax in [ax['panel_B'], ax['panel_B_sub'], ax['panel_C'], ax['panel_D']]:
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

    ax['panel_B'].set_xticks([])
    ax['panel_B_sub'].set_xticks([])
    ax['panel_B_sub'].axhline(0, color='k', alpha=0.2)
    ax['panel_C'].set_xticks([])
    ax['panel_C'].legend(frameon=False)
    ax['panel_C'].set_ylim(0, None)
    ax['panel_D'].set_xlabel("Trial", fontsize=label_size)
    ax['panel_D'].legend(frameon=False)
    ax['panel_D'].set_ylim(0, None)

    # fig.text(0.375, 0.38, "Logits", fontsize=label_size, ha='left', va='top', rotation='vertical')
    # fig.text(0.38, 0.65, r"$\Delta$perf.", fontsize=label_size-2, ha='left', va='top', rotation='vertical')
    # fig.text(0.375, 0.85, "Logit diff.", fontsize=label_size, ha='left', va='top', rotation='vertical')
    ax['panel_B'].set_ylabel(r"$\Delta$ hist. logits", fontsize=label_size)
    ax['panel_B_sub'].set_ylabel(r"$\Delta$ perf.", fontsize=label_size-2)
    ax['panel_C'].set_ylabel("Left logits", fontsize=label_size)
    ax['panel_D'].set_ylabel("Right logits", fontsize=label_size)

    labels = [
        {'label_text': 'a', 'xpos': 0.06, 'ypos': 0.87},
        {'label_text': 'b', 'xpos': 0.06, 'ypos': 0.585},
        {'label_text': 'c', 'xpos': 0.06, 'ypos': 0.46},
        {'label_text': 'd', 'xpos': 0.06, 'ypos': 0.27}
    ]
    for lbl in labels:
        fig.text(lbl['xpos'], lbl['ypos'], lbl['label_text'], fontsize=33, fontweight='bold')

    plt.savefig(f"figures/tiny_possibilities_{bt}.png", bbox_inches='tight', dpi=300)

    if bt == 0:
        plt.show()
    else:
        plt.close()