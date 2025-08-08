import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from load_network import loaded_network
import load_data
import seaborn as sns
from scipy.stats import sem

net_save = pickle.load(open("net_save", 'rb'))
dpi = None

plot_0_only = False

label_size = 24
ticksize = 18


result_folder2name = {"000/": "000",
                      "simple_filter/": "100",
                      "100-sv_reduced_ie_timeout_only/": "100-sv_timeout_only",
                      "100_double_decay/": "100_double_decay",
                      "static_action_deca_vector_plus_100/": "100-sv",
                      "100_double_sv/": "100_double_sv",
                    #   "results_indiv/": "Indiv. decay const.",
                    #   "encode/": "110",
                    #   "decay/": "101",
                    #   "infer_decay/": "100-is",
                    #   "limited_mecha_plus_lstm/": "Lim_Mecha+LSTM",
                    #   "enc_dec_20/": "Encode+Decay finite",
                    #   "mecha_sweep/": "111",
                    #   "contrast_only_mecha_plus_lstm/": "2-1-1",
                      "200-symmetric_base/": "200-symmetric_base",
                      "200_double_sv/": "200_double_sv",
                    #   "expo_filter+contrastLSTM/": "200",
                    #   "static_action_deca_vector_plus_200/": "Static decay vec.+200",
                    #   "simple_encode+contrastLSTM/": "21*0",
                    #   "limited_mecha_plus_lstm_enc/": "121",
                    #   "limited_mecha_plus_lstm_dec/": "112",
                    #   "limited_mecha_plus_lstm_enc_dec/": "122",
                    #   "201/": "201",
                    #   "210/": "210",
                    #   "202/": "202",
                    #   "reward_rate_scalar/": "reward_scalar",
                    #   "learned_reward_scalar/": "learned_reward_scalar",
                    #   "infer_deca_plus_200/": "200-is",
                    #   "simp_211_symm/": "simp_211_symm",
                    #   "limited_mecha_plus_lstm_cont/": "211",
                    #   "mecha_plus_lstm_simple_history/": "220",
                    #   "limited_mecha_plus_lstm_enc_cont/": "221",
                    #   "limited_mecha_plus_lstm_dec_cont/": "212",
                    #   "222-20/": "222-20",
                    #   "mecha_plus_sweep/": "222",
                    #   "LSTM_2D_contNdec/": "LSTM_2D_contNdec",
                    #   "LSTM_2D_contNenc/": "LSTM_2D_contNenc",
                    #   "LSTM_2D_all_20/": "LSTM_2D_all_20",
                    #   "LSTM2D-all-reduced_input_3/": "LSTM2D-all-reduced_input_3",
                    #   "LSTM2D-all-reduced_input_4/": "LSTM2D-all-reduced_input_4",
                    #   "LSTM_2D_all/": "LSTM_2D_all",
                    #  "222-2_reduc_mirror/": "222-2_reduc_mirror",
                    #   "222-2_mirror/": "222-2_mirror",
                    #   "202-20/": "202-20",
                    #   "200-iv_symmetric/": "200-iv_symmetric",
                    #   "temp_beta_gamma/": "200-iv_beta_gamma_derivative",
                    #   "bi_lstm_plus/": "bi_lstm_plus",
                    #   "mecha_plus_lstm_2/": "mecha_plus_lstm_dim2",
                    #   "infer_deca_vector_plus_200/": "Infer_Decay_Vector+200",
                      "simpler_211/": "simpler_211",
                      "simp_211_lstm_refit/": "simp_211_lstm_refit",
                      "new-211_reg/": "new-211_reg",
                      "210_twodec_rewonly_symm_noenc_regularised/": "definite_210_twodec_rewonly",
                      "210_twodec_allinp_symm_noenc_regularised/": "definite_210_twodec",
                    #   "simpler_222/": "simpler_222",
                      "200-iv_beta_gamma_derivative/": "200-iv_beta_gamma_derivative",
                      "infer_action_deca_vector_plus_200/": "200-iv",
                      "lstm_sweep/": "RNN"
                      }

final_net_save = pickle.load(open("net_save", 'rb'))


if True:
    fig = plt.figure(figsize=(14, 6.5))
    a0 = plt.gca()

    names = ['Base', 'abcd', 'abcD', 'abCd', 'abCD', 'aBcd', "Abcd", "aBCD", "AbCD", "ABcd", "ABcD", "ABCd", 'ABCD', 'ABCD reg.', 'A+ID', 'RNN']
    for i, name in enumerate(names):
        counter = 0
        for j in range(4 if name != 'A+ID' else 3): # lol
            if name not in final_net_save:
                continue
            label1 = "Test" if i == 0 and counter == 0 else None
            label2 = "Validation" if i == 0 and counter == 0 else None
            print(name)
            a0.plot(i + counter * 0.1 - 0.15, final_net_save[name][j][0], 'ko', label=label2)
            a0.plot(i + counter * 0.1 - 0.15, final_net_save[name][j][1], 'ro', label=label1)
            # a0.plot(i + counter * 0.1 - 0.15, net_save[name + "_{}".format(j)][2], 'ko', alpha=0.2, label=label2)
            counter += 1

            # if i == 0:
            #     color = 'k' if counter == 4 else 'white'
            #     plt.gca().annotate(
            #         'Different\nseeds',                      # Text to display
            #         xy=((counter-1) * 0.1 - 0.15 + i, net_save[name + "_{}".format(j)][1]),              # Point to annotate
            #         xytext=(0.1, 70.5),          # Position of the text
            #         fontsize=32,
            #         c = color,
            #         arrowprops=dict(facecolor='black', arrowstyle='->')  # Arrow properties
            #     )

    if not plot_0_only:
        a0.set_ylim(69, 72)
    # plt.plot([0 - 0.15, 0 + 0.15], [charles, charles], 'orange', alpha=0.6, label="Charles fit")
    # plt.plot([1 - 0.15, 1 + 0.15], [70.05652112793643, 70.05652112793643], 'green', alpha=0.6, label="LOO fit")
    # plt.plot([0 - 0.15, 0 + 0.15], [73.03353210220128, 73.03353210220128], 'blue', alpha=0.6, label="Session fit")

    a0.legend(fontsize=label_size-3, loc="lower right", frameon=False)
    a0.set_ylabel("Avg. trialwise model fit", size=label_size)

    names[0] = 'base'  # de-capitalise

    a0.set_xticks(range(len(names)), names, rotation=45, size=20, ha='right') # size=32
    a0.tick_params(axis='both', which='major', labelsize=ticksize)
    sns.despine()

    plt.tight_layout()
    plt.savefig("full_model_comp", dpi=150)
    plt.show()
    quit()



charles = pickle.load(open("charles_behavior_models/behavior_models/performance_list", 'rb'))
charles = np.concatenate(charles)
charles = 100 * np.exp(charles.mean())

names = result_folder2name.values()
yticks = []

fig, (a0, a2) = plt.subplots(1, 2, figsize=(14, 6.5), width_ratios=[1, 1.4])

names = ['abcd', "Abcd", "aBCD", "AbCD", "ABcD", "ABCd", 'ABCD', 'RNN']
for i, name in enumerate(names):
    counter = 0
    for j in range(4): # lol
        if name not in final_net_save:
            continue
        label1 = "Test" if i == 0 and counter == 0 else None
        label2 = "Validation" if i == 0 and counter == 0 else None
        print(name)
        a0.plot(i + counter * 0.1 - 0.15, final_net_save[name][j][0], 'ko', label=label2)
        a0.plot(i + counter * 0.1 - 0.15, final_net_save[name][j][1], 'ro', label=label1)
        # a0.plot(i + counter * 0.1 - 0.15, net_save[name + "_{}".format(j)][2], 'ko', alpha=0.2, label=label2)
        counter += 1

        if i == 1:
            print('arrows')
            print(((counter-1) * 0.1 - 0.15 + i, final_net_save[name][j][0]))
            color = 'k' if counter == 4 else 'white'
            a0.annotate(
                'Different\nseeds',                      # Text to display
                xy=((counter-1) * 0.1 - 0.15 + i, final_net_save[name][j][0]),              # Point to annotate
                xytext=(0.1, 71.5),          # Position of the text
                fontsize=28,
                c = color,
                arrowprops=dict(facecolor='black', arrowstyle='->')  # Arrow properties
            )

if not plot_0_only:
    a0.set_ylim(69.4, 72)
# plt.plot([0 - 0.15, 0 + 0.15], [charles, charles], 'orange', alpha=0.6, label="Charles fit")
# plt.plot([1 - 0.15, 1 + 0.15], [70.05652112793643, 70.05652112793643], 'green', alpha=0.6, label="LOO fit")
# plt.plot([0 - 0.15, 0 + 0.15], [73.03353210220128, 73.03353210220128], 'blue', alpha=0.6, label="Session fit")

a0.legend(fontsize=label_size-3, loc="lower right", frameon=False)
a0.set_ylabel("Avg. trialwise model fit", size=label_size)

# names[0] = 'base'  # de-capitalise

a0.set_xticks(range(len(names)), names, rotation=45, size=20, ha='right') # size=32
a0.tick_params(axis='both', which='major', labelsize=ticksize)
sns.despine()

# plt.tight_layout()
# plt.savefig("model comp neuromonster", dpi=dpi)
# plt.show()


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

eval_on_heldout = True

if eval_on_heldout:
    input_seq, mask_to_use, biased_blocks = load_data.gib_test_data()


bias_diff = np.abs(biased_blocks[:, :-1] - biased_blocks[:, 1:])
before, after = 8, 20
specific_contrast = [0, 0.0625, 0.125, 0.25, 1][0]


take = [0, 1, 2, 3, 4]
files = [f for i, f in enumerate(['./final_nets/final_net_save_88617248.p', './final_nets/final_net_save_82072961.p',
                                  './final_nets/final_net_save_84553697.p', "./210_twodec_rewonly_symm_noenc_regularised//simple_decay_infer_save_76678850.p",
                                  'lstm_sweep/lstm_sweep_save_98989982.p']) if i in take]
names = [f for i, f in enumerate(["abcd", "aBCD", "Abcd", "ABCD", "RNN"]) if i in take]
colors = [f for i, f in enumerate(["C0", "C4", "C1", "C2", "C3"]) if i in take]

temp_save_title = f"res level {len(take)}"

session_end = False
n_from_end = 50

performance_mice = []
performance_mice_0 = [[] for _ in range(before + after)]
# block switch plots for only 0 contrasts
if True:
    for k, (file, name, color) in enumerate(zip(reversed(files), reversed(names), reversed(colors))):
        contrast_perf = [[] for _ in range(before + after)]
        contrast_counter = np.zeros(before + after)

        infos = pickle.load(open(file, 'rb'))

        if name == "111":
            ind = 19180
            nll = 70.8884806443464,  # saved values since this thing has no best net
            intermediates = pickle.load(open("./best_nets_params/" + "mecha_sweep_save_6738724.p"[:-2] + "_intermediate.p", 'rb'))
            infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
            reload_net = loaded_network(infos, use_best=False) 
        elif name == "LSTM" or name == "RNN":
            ind = 16100
            nll = 71.88036007212331  # saved values since this thing has no best net
            intermediates = pickle.load(open("./best_nets_params/" + "lstm_sweep_save_98989982.p"[:-2] + "_intermediate.p", 'rb'))
            infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
            reload_net = loaded_network(infos, use_best=False)
        else:
            nll = 100 * np.exp(- infos['best_test_nll'])
            reload_net = loaded_network(infos, use_best=True)

        performance = reload_net.return_predictions(input_seq, action_subsetting=True)

        print(nll)
        print(100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
        assert np.allclose(nll, 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

        if name == "":
            continue

        if session_end:
            performance_period = []
            for i, session_mask in enumerate(mask_to_use):
                performance_period.append(performance[i, session_mask.sum() - n_from_end : session_mask.sum()])
            performance_period = np.array(performance_period)
        else:
            performance_period = []
            for i, session_mask in enumerate(mask_to_use):
                block_switches = np.where(np.isclose(bias_diff[i], 0.6))[0]
                for spot in block_switches:
                    spot = spot + 1  # actualy block switch is one trial later
                    if session_mask[spot - before:spot + after].sum() != before + after:
                        continue
                    performance_period.append(performance[i, spot - before:spot + after])

                    for j in range(before + after):
                        if input_seq[i, spot - before + j, 3] == 0 and input_seq[i, spot - before + j, 4] == 0:
                            contrast_perf[j].append(performance[i, spot - before + j])  # +1 since rewards, which we are pulling, are delayed one entry

                    if k == 0:
                        performance_mice.append((input_seq[i, spot - before + 1:spot + after + 1, 5] + 1) / 2)  # +1 since rewards, which we are pulling, are delayed one entry
                        for j in range(before + after):
                            if input_seq[i, spot - before + j, 3] == 0 and input_seq[i, spot - before + j, 4] == 0:
                                performance_mice_0[j].append((input_seq[i, spot - before + j + 1, 5] + 1) / 2)  # +1 since rewards, which we are pulling, are delayed one entry

            performance_period = np.array(performance_period)

        if plot_0_only:
            means = [np.mean(contrast_perf[i]) for i in range(before + after)]
            sems = np.array([sem(contrast_perf[i]) for i in range(before + after)])
            a2.errorbar(np.arange(-before, after), means, sems / 2, label=name, color=color)
        else:
            print(performance_period.shape)
            if session_end:
                a2.errorbar(np.arange(-n_from_end, 0), performance_period.mean(0), sem(performance_period, axis=0) / 2, label=name, color=color)
            else:
                if name != 'ABCD':
                    a2.errorbar(np.arange(-before, after), performance_period.mean(0), sem(performance_period, axis=0) / 2, label=name, color=color)
                else:
                    print('yoyo')
                    a2.errorbar(np.arange(-before, after), performance_period.mean(0), sem(performance_period, axis=0) / 2, label=name, color=color, lw=2)

    if session_end:
        a2.set_xlabel("Trials from session end", size=label_size)
        a2.set_xlim(-n_from_end, 0)
    else:
        a2.set_xlabel("Trials around block switch", size=label_size)
        a2.axvline(- 0.5, c='k', ls='--')  # put line just before the first trial of new block
        a2.set_xlim(-before, after)
        # a2.set_ylim(0.68, 0.83)
    prefix = "Train set " if training_data else "Val. set "
    if eval_on_heldout:
        prefix = "Test set "
    a2.set_ylabel(prefix + "model fit", size=label_size)
    a2.tick_params(axis='both', which='major', labelsize=ticksize)

    if not plot_0_only:
        a2.legend(frameon=False, loc=4, fontsize=label_size-2)
    else:
        a2.legend(frameon=False, loc="upper center", fontsize=label_size-2)
    sns.despine()

# performance_mice = np.array(performance_mice)
# if not plot_0_only:
#     a1.plot(np.arange(-before, after), performance_mice.mean(0), color='k')
#     a1.fill_between(np.arange(-before, after), performance_mice.mean(0) - sem(performance_mice, axis=0) / 2, performance_mice.mean(0) + sem(performance_mice, axis=0) / 2, color='k', alpha=0.2)
# else:
#     means = [np.mean(performance_mice_0[i]) for i in range(before + after)]
#     sems = [sem(performance_mice_0[i]) for i in range(before + after)]
#     a1.plot(np.arange(-before, after), means, color='k')
#     a1.fill_between(np.arange(-before, after), np.array(means) - np.array(sems) / 2, np.array(means) + np.array(sems) / 2, color='k', alpha=0.2)
# a1.axvline(- 0.5, c='k', ls='--')  # put line just before the first trial of new block
# a1.set_xlim(-before, after)
# a1.set_xlabel("Trials around block switch", size=label_size)
# a1.set_ylabel("Mouse performance", size=label_size)
# a1.tick_params(axis='both', which='major', labelsize=ticksize)

# a1.set_xticks([-5, 0, 5, 10, 15, 20], [-5, 0, 5, 10, 15, 20])

a0.text(-0.16, 1.08, "a", transform=a0.transAxes, fontsize=47, va='top', ha='right')
# a1.text(-0.15, 1.08, "b", transform=a1.transAxes, fontsize=47, va='top', ha='right')
a2.text(-0.08, 1.08, "b", transform=a2.transAxes, fontsize=47, va='top', ha='right')

# if not plot_0_only:
#     a1.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])

# if not plot_0_only:
#     a1.set_ylim(0.7, 0.9)
#     a1.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])
# else:
#     # a2.set_ylim(0.3, 0.75)
#     pass

sns.despine()


plt.tight_layout()
if session_end:
    plt.savefig("session end comp", dpi=300)
else:
    plt.savefig(temp_save_title + f"_0_only_{plot_0_only}", dpi=300)
plt.show()

quit()
files = [("save_61912285.p", "save_61912285_step_18600.p"), ("mecha_sweep_save_6738724.p", "mecha_sweep_save_6738724_intermediate.p"),
         ("save_61009871.p", "save_61009871_intermediate.p"), ("save_5144828.p", "save_5144828_intermediate.p")]
names = ["Exp. filter", "Mechanical memory", "Bi-RNN", "RNN"]
nlls = [69.6208, 70.8884806443464, 71.32133763671284, 71.57461094053133]
indices = [18600, 19180, 74480, 74480]

performances = []
reduced_names = []
for file, name, nll, ind in zip(files, names, nlls, indices):

    if name in ["Mechanical memory", "Bi-RNN"]:
        continue

    if name != "Exp. filter":
        intermediates = pickle.load(open("./best_nets_params/" + file[1], 'rb'))

        infos = pickle.load(open("./best_nets_params/" + file[0], 'rb'))

        if 'params_list' in infos:
            intermediates['params_list'] = infos['params_list']

        infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    else:
        intermediates = pickle.load(open("./best_nets_params/" + "save_61912285_step_18600.p", 'rb'))
        infos = pickle.load(open("./best_nets_params/" + "save_61912285.p", 'rb'))
        infos['params_lstm'] = intermediates['params_lstm']


    reload_net = loaded_network(infos)
    performance = reload_net.return_predictions(input_seq, action_subsetting=True)

    assert np.allclose(nll, 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    print(nll)
    print(100 * np.exp(np.log(performance[mask_to_use]).mean()))

    if name == "":
        continue

    performances.append(performance)
    reduced_names.append(name)

diffs = []
for i, session_mask in enumerate(mask_to_use):
    block_switches = np.where(np.isclose(bias_diff[i], 0.6))[0]
    for spot in block_switches:
        spot = spot + 1  # actualy block switch is one trial later
        if session_mask[spot - before:spot + after].sum() != before + after:
            continue
        # for performance, name in zip(performances, reduced_names):
        #     plt.plot(np.arange(-before, after), performance[i, spot - before:spot + after], label=name)

        diffs.append(performances[0][i, spot - before:spot + after] - performances[1][i, spot - before:spot + after])

        # plt.axvline(- 0.5, c='k', ls='--')  # put line just before the first trial of new block
        # plt.xlim(-before, after)
        # plt.xlabel("Trials around block switch", size=label_size)
        # prefix = "Train set " if training_data else "Test data "
        # plt.ylabel(prefix + "predictive performance", size=label_size)
        # plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.legend(frameon=True, loc=4, fontsize=label_size-3)
        # sns.despine()
        # plt.tight_layout()
        # plt.show()
