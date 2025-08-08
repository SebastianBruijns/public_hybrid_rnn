import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_network import loaded_network
import load_data

net_nlls = {}
net_train_nlls = {}
net_train_opt_test_nlls = {}
net_train_opt_train_nlls = {}
net_ids = {}

index_train_opt = {}

compute_0_performance = False

# lstm_contrast_scalar = desired_LSTM_contrast
# two_decays = desired_two_histories
# decay_input = [0, 1, 2] if desired_action_dep_decay else [1]
# decay_addon = [3] if desired_rew_dep_decay else []

def name_converter(tag):
    name = ""
    
    name += "A" if tag[0] else "a"
    name += "B" if tag[1] else "b"
    name += "C" if tag[2] else "c"
    name += "D" if tag[3] else "d"
    return name

final_net_save = pickle.load(open("net_save", 'rb'))

_, _, test_input_seq, test_test_mask = load_data.gib_data_fast()
input_seq_val, validation_mask = load_data.gib_test_data()


if False:
    for i, file in enumerate(os.listdir("final_nets")):

        sweep_params = ['seed', 'train_batch_size', 'n_hiddens', 'n_contrast_hiddens', 'learning_rate', 'weight_decay']

        infos = pickle.load(open("./final_nets/" + file, 'rb'))

        if infos['lstm_contrast_scalar'] and (infos['input_list'] == [0, 1, 2, 5, 3, 4]):
            continue

        df = infos['all_scalars']

        min_index = df[df.step % 200 == 0].test_nll.idxmin()


        train_min_index = df[df.step % 1000 == 0].train_nll.idxmin()

        identifier = tuple(infos[param] for param in sweep_params if param in infos)
        tag = (infos['lstm_contrast_scalar'], infos['two_decays'], infos['decay_input'] == [0, 1, 2], infos['decay_addon'] == [3])

        if tag not in net_nlls:
            net_nlls[tag] = []
            net_ids[tag] = []
            net_train_opt_test_nlls[tag] = []
            net_train_nlls[tag] = []
            net_train_opt_train_nlls[tag] = []
            index_train_opt[tag] = []

        net_nlls[tag].append(100 * np.exp(- infos['all_scalars'].test_nll[min_index]))
        net_train_nlls[tag].append(100 * np.exp(- infos['all_scalars'].train_nll[min_index]))
        net_train_opt_test_nlls[tag].append(100 * np.exp(- infos['all_scalars'].test_nll[train_min_index]))
        net_train_opt_train_nlls[tag].append(100 * np.exp(- infos['all_scalars'].train_nll[train_min_index]))
        net_ids[tag].append((identifier, file))

        index_train_opt[tag].append(train_min_index)

        # sweep_params = list(product([100, 101, 102, 103], [8, 64, 16], [16, 8], [16, 8], [1e-3], [1e-4, 1e-3], combs)) # [0.1, 1, 10, 100]
        # seed, train_batch_size, n_hiddens, n_contrast_hiddens, learning_rate, weight_decay, setting = sweep_params[int(sys.argv[1])]

    sorted_keys = sorted(net_nlls, key=lambda k: max(net_nlls[k]))

    plt.figure(figsize=(12, 6))

    # delete the (False, True, False, True) entry from sorted_keys
    sorted_keys = [key for key in sorted_keys if key != (False, True, False, True)]

    if compute_0_performance:
        pass
    _, _, test_input_seq, test_test_mask = load_data.gib_data_fast()

    input_seq_val, validation_mask = load_data.gib_test_data()

    for tag_counter, tag in enumerate(sorted_keys):
        sorted_ids = np.argsort(net_nlls[tag])
        best_index = sorted_ids[-1]
        best_id = net_ids[tag][best_index]

        print(f"4 seeds of best {tag} net")
        # net_save = pickle.load(open("net_save", 'rb'))
        counter = 1
        for i in sorted_ids:
            match = True
            for j in range(1, len(best_id[0])):
                match = match and (net_ids[tag][i][0][j] == best_id[0][j])
            if match:
                print("{}, {:.3f}, {:.3f}, best train index: (not currently) (training optimised: {:.3f}, {:.3f})".format(net_ids[tag][i], net_nlls[tag][i], net_train_nlls[tag][i], net_train_opt_test_nlls[tag][i], net_train_opt_train_nlls[tag][i]))
                # net_save[result_folder2name[result_folder] + "_{}".format(counter+1)] = (ids[i], nll_list[i], train_nll[i], raw_nll[i], train_opt_nll_list[i])
                counter += 1
                plt.plot([tag_counter - 0.15 + (counter - 2) * 0.1], [net_nlls[tag][i]], 'ko')

                # compute test (validation) performance
                infos = pickle.load(open("final_nets/" + net_ids[tag][i][-1], 'rb'))
                reload_net = loaded_network(infos)
                assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

                # intermediate_params = pickle.load(open("final_nets_params/" + net_ids[tag][i][-1][:-2] + "_intermediate.p", 'rb'))
                # reload_net.params = intermediate_params["params_list"][index_train_opt[tag][i] * 20 // 1000]
                # assert np.allclose(net_train_opt_test_nlls[tag][i], 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

                val_perf = 100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())
                print(index_train_opt[tag][i] * 20 // 1000, val_perf)
                plt.plot([tag_counter - 0.15 + (counter - 2) * 0.1], [val_perf], 'ro')

                if name_converter(tag) not in final_net_save:
                    final_net_save[name_converter(tag)] = []
                final_net_save[name_converter(tag)].append((net_nlls[tag][i], val_perf))

                # if net_ids[tag][i][-1] == "final_net_save_65945835.p":
                #     quit()

                if compute_0_performance:
                    infos = pickle.load(open("final_nets/" + net_ids[tag][i][-1], 'rb'))

                    reload_net = loaded_network(infos)
                    print(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
                    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
                    
                    zero_performance = 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[np.logical_and(test_test_mask, (test_input_seq[:, :-1, 3] + test_input_seq[:, :-1, 4]) == 0)]).mean())
                    print(zero_performance)

                    # net_save[result_folder2name[result_folder] + "_zeros_{}".format(counter+1)] = zero_performance

        print()
        # print(tag, net_nlls[tag][best_index], net_train_opt_test_nlls[tag][best_index], net_ids[tag][best_index])

    # quit()

for file in ['infer_decay_save_65551925.p', 'infer_decay_save_86847687.p', 'infer_decay_save_7248438.p']:
    infos = pickle.load(open("./200-iv_beta_derivative/" + file, 'rb'))
    reload_net = loaded_network(infos)
    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    if "A+ID" not in final_net_save:
        final_net_save["A+ID"] = []
    print("A+ID")
    print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))
    final_net_save["A+ID"].append((100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()), 100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())))

pickle.dump(final_net_save, open("net_save", 'wb'))

quit()

for file, ind, nll in zip(["mecha_plus_sweep_save_73377761.p", "mecha_plus_sweep_save_18650140.p", "mecha_plus_sweep_save_2968296.p", "mecha_plus_sweep_save_95974159.p"],
                          [1970, 4070, 4450, 2650],
                          [69.484, 69.484, 69.485, 69.487]):
    folder = "000/"
    infos = pickle.load(open(folder + file, 'rb'))
    intermediates = pickle.load(open("./best_nets_params/" + file[:-2] + "_intermediate.p", 'rb'))
    infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    reload_net = loaded_network(infos, use_best=False)
    if "Base" not in final_net_save:
        final_net_save["Base"] = []
    print("Base")
    print(nll, 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))
    quit()
    final_net_save["Base"].append((100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()), 100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())))
# pickle.dump(final_net_save, open("net_save", 'wb'))

quit()

for file, ind, nll in zip(["lstm_sweep_save_38821501.p", "lstm_sweep_save_16089721.p", "lstm_sweep_save_42805248.p", 'lstm_sweep_save_98989982.p'],
                          [79520, 13120, 73920, 16100],
                          [71.785, 71.839, 71.856, 71.88036007212331]):
    folder = "lstm_sweep/"
    infos = pickle.load(open(folder + file, 'rb'))
    intermediates = pickle.load(open("./best_nets_params/" + file[:-2] + "_intermediate.p", 'rb'))
    infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    reload_net = loaded_network(infos, use_best=False)
    if "RNN" not in final_net_save:
        final_net_save["RNN"] = []
    print("RNN")
    print(nll, 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))
    final_net_save["RNN"].append((100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()), 100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())))

print()

# plt.plot([tag_counter + 2], [nll], 'ko')
# plt.plot([tag_counter + 2], [100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())], 'ro')

# print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))

# quit()

for file in ['simple_decay_infer_save_9114044.p', 'simple_decay_infer_save_62867199.p', 'simple_decay_infer_save_77872186.p', "simple_decay_infer_save_76678850.p"]:
    # file = "simple_decay_infer_save_16844694.p" # high beta
    infos = pickle.load(open("./210_twodec_rewonly_symm_noenc_regularised//" + file, 'rb'))
    reload_net = loaded_network(infos)
    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    if "ABCD reg." not in final_net_save:
        final_net_save["ABCD reg."] = []
    print("ABCD reg")
    print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))
    final_net_save["ABCD reg."].append((100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()), 100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())))
# plt.plot([tag_counter + 1], [100 * np.exp(-infos['best_test_nll'])], 'ko')
# plt.plot([tag_counter + 1], [100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean())], 'ro')

# print(100 * np.exp(np.log(reload_net.return_predictions(input_seq_val, action_subsetting=True)[validation_mask]).mean()))

pickle.dump(final_net_save, open("net_save", 'wb'))

# plt.gca().set_xticks(range(len(sorted_keys)), [name_converter(t) for t in sorted_keys], rotation=45, size=18, ha='right') # size=32
# plt.gca().set_xticks(range(len(sorted_keys) + 2), [tuple(int(b) for b in t) for t in sorted_keys] + ["ABCD_reg", "RNN"], rotation=45, size=18, ha='right') # size=32
# plt.tight_layout()
# sns.despine()
# plt.savefig("final_leaderboard_validation_opt.png")
# plt.show()