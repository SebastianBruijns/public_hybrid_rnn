import matplotlib.pyplot as plt
import pickle
import load_data
import numpy as np
import jax.numpy as jnp
import haiku as hk
import pickle
from final_net import Final_net
from simple_decay import Simple_Infer_decay
import jax


np.set_printoptions(suppress=True)

n_choices = load_data.n_choices

from run_rnn_functions import network_type_to_params

network_type2class = {"<class '__main__.Simple_Infer_decay'>": Simple_Infer_decay,
                      "<class 'simple_decay_infer.Simple_Infer_decay'>": Simple_Infer_decay,
                      "<class '__main__.Final_net'>": Final_net}

contrasts = [np.array([1, 0]),
             np.array([0.25, 0]),
             np.array([0.125, 0]),
             np.array([0.0625, 0]),
             np.array([0, 0]),
             np.array([0, 0.0625]),
             np.array([0, 0.125]),
             np.array([0, 0.25]),
             np.array([0, 1])]

ex_contra = [np.array([1, 0]),  # the extended set of contrasts
             np.array([0.5, 0]),
             np.array([0.25, 0]),
             np.array([0.125, 0]),
             np.array([0.0625, 0]),
             np.array([0, 0]),
             np.array([0, 0.0625]),
             np.array([0, 0.125]),
             np.array([0, 0.25]),
             np.array([0, 0.5]),
             np.array([0, 1])]


class loaded_network:

    def __init__(self, infos, use_best=True):
        """
            model_keyword: model setup, whether to use memory, how many hidden units, etc, recover from loaded network
            params: fitted network weights
        """

        self.agent_class = network_type2class[infos["agent_class"]]
        self.params = infos['params_lstm'] if not use_best else infos['best_net']  # TODO: this would be good to use, but a lot of the old code is not set up for it
        # select the necessary kewords out of the infos dict, for the chosen agent_class

        self.model_keywords = {key: infos[key] for key in network_type_to_params[infos['agent_class']] if key in infos}
        self.input_list = infos['input_list']
        self.infos = infos

        if self.agent_class == Simple_Infer_decay:
            if 'mirror_enc' not in self.infos or not self.infos['mirror_enc']:
                self.input_matrix_1, self.input_bias_1 = self.params['simple__infer_decay/~_encoding_network/linear_1']['w'], self.params['simple__infer_decay/~_encoding_network/linear_1']['b']
                self.input_matrix_2, self.input_bias_2 = self.params['simple__infer_decay/~_encoding_network/linear']['w'], self.params['simple__infer_decay/~_encoding_network/linear']['b']
            if 'mirror_dec' not in self.infos or not self.infos['mirror_dec']:
                self.decay_matrix_1, self.decay_bias_1 = self.params['simple__infer_decay/~_decay_network/linear_1']['w'], self.params['simple__infer_decay/~_decay_network/linear_1']['b']
                self.decay_matrix_2, self.decay_bias_2 = self.params['simple__infer_decay/~_decay_network/linear']['w'], self.params['simple__infer_decay/~_decay_network/linear']['b']

            if 'symmetric_contrast_net' not in self.infos or not self.infos['symmetric_contrast_net']:
                self.contrast_matrix_1, self.contrast_bias_1 = self.params['simple__infer_decay/~_contrast_mlp/linear']['w'], self.params['simple__infer_decay/~_contrast_mlp/linear']['b']
                self.contrast_matrix_2, self.contrast_bias_2 = self.params['simple__infer_decay/~_contrast_mlp/linear_1']['w'], self.params['simple__infer_decay/~_contrast_mlp/linear_1']['b']

        def _lstm_fn(input_seq, return_all_states=True):
            """Cognitive models function."""

            self.model = self.agent_class(**self.model_keywords)

            batch_size = len(input_seq)
            initial_state = self.model.initial_state(batch_size)

            return hk.dynamic_unroll(
                self.model,
                input_seq,
                initial_state,
                time_major=False,
                return_all_states=return_all_states)
        self.lstm_fn = hk.transform(_lstm_fn)

    def return_predictions(self, input_seq, action_subsetting=False):
        """Cross-entropy loss between model-predicted and input behavior."""

        action_probs_seq, _ = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)

        if action_subsetting:
            return (action_probs_seq * action_seq).sum(-1)
        else:
            return action_probs_seq

    def nll_fn_lstm(self, input_seq, length_mask, beta=0):
        """Cross-entropy loss between model-predicted and input behavior."""

        action_probs_seq, stuff = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
        contrast_motivation, decay_vectors = stuff[-1]
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions

        contrast_motivation = jnp.diff(jnp.squeeze(contrast_motivation[1:]))  # compute first derivative
        decay_vectors = jnp.diff(decay_vectors[1:], axis=1)  # compute first derivative

        # jax.debug.print("normal loss {x}, contrast norm {y}", x=-jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum(), y=beta * jnp.sum(jnp.square(jnp.where(length_mask, contrast_motivation, 0))) / length_mask.sum())
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        return nll

    def nll_fn_lstm_with_var(self, input_seq, length_mask):
        """Same as above, but with NLL computation for every session individually as well.
            I wasn't able to get this done via just a bool flag, jax complained"""

        action_probs_seq, _ = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (actions were all shifted backwards)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        session_nlls = []
        for i in range(logs.shape[0]):
            session_nlls.append(np.mean(logs[i][length_mask[i]]))
        session_nlls = np.array(session_nlls)

        return nll, session_nlls

    # encoding of the last history input
    def input_process(self, input, addon=None):
        if 'mirror_enc' in self.infos and self.infos['mirror_enc']:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['simple__infer_decay/~_encoding_network/mirror_invariant_network/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['simple__infer_decay/~_encoding_network/mirror_invariant_network/~/linear_1']
                                    }, input, addon=addon)
        
        if ('network_memory' in self.infos and self.infos['network_memory']) or ('action_encoding' in self.infos and self.infos['action_encoding']):
            if 'Mirror_nets' in self.infos and self.infos['Mirror_nets']:
                from my_module import mirror_invariant_network
                network = hk.without_apply_rng(hk.transform(mirror_invariant_network()))
                return network.apply({'mirror_invariant_network/~/linear': self.params['mirror_mecha_plus/~_habit_manipulator/mirror_invariant_network/~/linear'],
                                      'mirror_invariant_network/~/linear_1': self.params['mirror_mecha_plus/~_habit_manipulator/mirror_invariant_network/~/linear_1']
                                      }, input, addon=addon)
            else:
                temp = (input[:, None] * self.input_matrix_2).sum(0) + self.input_bias_2
                new = np.tanh(temp)
                return (new[:, None] * self.input_matrix_1).sum(0) + self.input_bias_1
        else:
            return input
        
    def input_process_2(self, input, addon=None):
        if not self.infos['two_decays']:
            print("Error: this function is only for the two-decay networks")
            return None
        if 'mirror_enc' in self.infos and self.infos['mirror_enc']:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['simple__infer_decay/~_encoding_network/mirror_invariant_network_1/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['simple__infer_decay/~_encoding_network/mirror_invariant_network_1/~/linear_1']
                                    }, input, addon=addon)
        else:
            return input
        
    def decay_process(self, input, addon=None):  # for the special new nets which spit out a decay vector
        if 'mirror_dec' in self.infos and self.infos['mirror_dec']:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['simple__infer_decay/~_decay_network/mirror_invariant_network/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['simple__infer_decay/~_decay_network/mirror_invariant_network/~/linear_1']
                                    }, input, addon=addon)
        else:
            temp = (input[:, None] * self.decay_matrix_2).sum(0) + self.decay_bias_2
            new = np.tanh(temp)
            return (new[:, None] * self.decay_matrix_1).sum(0) + self.decay_bias_1

    def final_decay_1(self, input, addon=None):
        from my_module import MirrorInvariantNetwork
        def mirror_invariant_network(net_input, addon=None):
            model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
            return model(net_input, addon)
        network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
        return network.apply({'mirror_invariant_network/~/linear': self.params['final_net/~_decay_network/mirror_invariant_network/~/linear'],
                                'mirror_invariant_network/~/linear_1': self.params['final_net/~_decay_network/mirror_invariant_network/~/linear_1']
                                }, input, addon=addon)  

    def final_decay_2(self, input, addon=None):
        from my_module import MirrorInvariantNetwork
        def mirror_invariant_network(net_input, addon=None):
            model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
            return model(net_input, addon)
        network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
        return network.apply({'mirror_invariant_network/~/linear': self.params['final_net/~_decay_network/mirror_invariant_network_1/~/linear'],
                                'mirror_invariant_network/~/linear_1': self.params['final_net/~_decay_network/mirror_invariant_network_1/~/linear_1']
                                }, input, addon=addon)  

        
    def decay_process_2(self, input, addon=None):  # for the special new nets which spit out a decay vector
        if not self.infos['two_decays']:
            print("Error: this function is only for the two-decay networks")
            return None
        if 'mirror_dec' in self.infos and self.infos['mirror_dec']:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['simple__infer_decay/~_decay_network/mirror_invariant_network_1/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['simple__infer_decay/~_decay_network/mirror_invariant_network_1/~/linear_1']
                                    }, input, addon=addon)
        else:
            return None

    def process(self, action):
        temp = (action[:, None] * self.decay_matrix_2).sum(0) + self.decay_bias_2
        new = np.tanh(temp)
        return (new[:, None] * self.decay_matrix_1).sum(0) + self.decay_bias_1
        
    def mult_apply(self, action, n):
        if self.infos['network_decay']:
            for i in range(n):
                action = self.process(action)
            return action
        else:
            return np.exp(- self.self.params['mecha_history']['decay']) ** n * action * self.self.params['mecha_history']['history_weighting']

    def mult_apply_augmented(self, action, augmenter, n):
        for i in range(n):
            action = self.process(np.append(action, augmenter))
        return action
         

    # # other way around here, because matrices are used in opposite order for decay
    def cont_process(self, contrast, addon=None):
        if self.agent_class == Final_net:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_contrast_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['final_net/mirror_invariant_network/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['final_net/mirror_invariant_network/~/linear_1']
                                    }, contrast.reshape(1, -1), addon=addon.reshape(1, -1) if addon is not None else None)[0]
        if 'symmetric_contrast_net' in self.infos and self.infos['symmetric_contrast_net']:
            from my_module import MirrorInvariantNetwork
            def mirror_invariant_network(net_input, addon=None):
                model = MirrorInvariantNetwork(n_hiddens=self.infos['n_contrast_hiddens'], n_actions=3)
                return model(net_input, addon)
            network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
            return network.apply({'mirror_invariant_network/~/linear': self.params['simple__infer_decay/mirror_invariant_network/~/linear'],
                                    'mirror_invariant_network/~/linear_1': self.params['simple__infer_decay/mirror_invariant_network/~/linear_1']
                                    }, contrast, addon=addon)
        else:
            temp = (contrast[:, None] * self.contrast_matrix_1).sum(0) + self.contrast_bias_1
            new = np.tanh(temp)
            return (new[:, None] * self.contrast_matrix_2).sum(0) + self.contrast_bias_2
    
    def manual_predict(self, trial, input_seq, s=0):
        total = np.zeros(3)
        for i in range(self.infos['history_slots']):
            history = self.mult_apply(np.zeros(3), trial + 1) if (trial - i) < 0 else \
                    self.mult_apply(self.input_process(input_seq[s, trial - i, self.infos['input_list'][:-2]]), i)
            total += history
        energies = total + self.cont_process(input_seq[s, trial, self.infos['input_list']][-2:])
        probs = np.exp(energies) / np.exp(energies).sum()
        action_probs = (1 - 1e-5) * probs + 1e-5 / 3  # fudging for non-zero probs
        return action_probs, total, energies

    def get_baseline(self):
        zero_energy = self.cont_process(np.array([0, 0]))
        return zero_energy[0] - zero_energy[2]

    def plot_pmf_energies(self, show=True, use_extended=False):
        pmf_energies = np.zeros(len(contrasts) if not use_extended else len(ex_contra))
        for i, c in enumerate(contrasts if not use_extended else ex_contra):
            activities = self.cont_process(c)
            pmf_energies[i] = activities[0] - activities[2]

        print(pmf_energies)
        if self.agent_class == BiRNN:
            pmf_energies = 0.5 * pmf_energies
        if show:
            plt.plot(pmf_energies)
            plt.axhline(0, c='k')
            plt.axvline(4, c='k')
        
            plt.show() # TODO: only plot if asked, not only show if asked

        return pmf_energies

if __name__ == "__main__":
    file = 'mirror_mecha_plus_save_18756433.p'
    infos = pickle.load(open("./222-2_mirror/" + file, 'rb'))
    reload_net = loaded_network(infos)

    input_seq, train_mask, input_seq_test, test_mask, _, _ = load_data.gib_data(file="./processed_data/all_mice.csv")

    print(infos['file'])
    print(infos['train_nll_lstm'])
    print(infos['test_nll_lstm'])
    print(infos['n_training_steps'])

    probs = reload_net.return_predictions(input_seq[:1, :25])

    print(probs[0][:25])
    print(input_seq[0, :25])
    
    quit()
    print(reload_net.nll_fn_lstm(input_seq, train_mask))

    def create_exp_filter(decay, length):
        weights = jnp.exp(- decay * jnp.arange(length))
        weights /= weights.sum()
        return weights

    # some code for looking at model training?
    fs = 20
    df = infos['all_scalars']
    plt.figure(figsize=(16, 9))
    plt.plot(df.step, 100 * np.exp(- df.train_nll), label='Train set')
    plt.plot(df.step, 100 * np.exp(- df.test_nll), label='Test set')
    plt.axhline(70.393, c='k', label="GLM test")
    plt.axhline(72.1110463142395, c='g', label="LSTM test")
    plt.axhline(71.92094922065735, c='r', label="BiLSTM test")
    plt.axhline(73.34137, ls='--', c='k', label="Theoretical limit")
    plt.xlim(1000, None)
    plt.ylim(66, None)
    plt.xlabel("Training step", size=fs+4)
    plt.ylabel("Prediction accuracy in %", size=fs+4)
    plt.title(infos['file'] + ' ' + infos['agent_class'], size=fs)
    plt.legend(frameon=False, fontsize=fs)
    plt.tight_layout()
    # plt.savefig(infos['file'] + ' ' + infos['agent_class'] + '.png')
    plt.show()

