"""
    General functions for fitting and evaluating RNN.

    https://github.com/google-deepmind/dm-haiku#why-haiku: Since the actual computation performed by our loss function doesn't rely on random numbers, we pass in None for the rng argument.
    TODO: cut off first trial?
    DONE: If I pass the feedback at the wrong spot, the model should be perfect (modulo 0 contrasts), cause the feedback tells it how the mouse answers -> Tested accidentaly, in fact true
    TODO: Design pickle saving nicer (don't overwrite)
    TODO: save all network params, not just the one in kwargs, so we can survive a default param change
"""
import optax
import matplotlib.pyplot as plt
import load_data
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import haiku as hk
import pickle
import time
from final_net import default_params as params_final_net
from simple_decay import default_params as params_simple_decay

n_choices = load_data.n_choices

file_prefix = ['.', '/usr/src/app'][0]

network_type_to_params = {"<class '__main__.Simple_Infer_decay'>": params_simple_decay,
                          "<class '__main__.Final_net'>": params_final_net}


network_type_to_name = {"<class 'mirrored_mecha_plus.Simple_Infer_decay'>": 'simple_decay_infer',
                        "<class '__main__.Simple_Infer_decay'>": 'simple_decay_infer',
                        "<class '__main__.Final_net'>": 'final_net',}



# @profile
def initialise_and_train(agent_class, input_list, n_training_steps=5000, learning_rate=1e-3, weight_decay=1e-4, seed=4, train_batch_size=None,
                         file=file_prefix + "/processed_data/all_mice.csv", eval_every=20, save_info=False, beta=0, gamma=0, delta=0, **kwargs):

    if file == file_prefix + "/processed_data/all_mice.csv":
        input_seq, train_mask, input_seq_test, test_mask = load_data.gib_data_fast()
    else:
        print(file)
        input_seq, train_mask, input_seq_test, test_mask, _, _ = load_data.gib_data(file=file)

    rng_seq = hk.PRNGSequence(seed)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    init_opt = jax.jit(optimizer.init)

    def _lstm_fn(input_seq, return_all_states=True):
        """Cognitive models function."""

        model = agent_class(**kwargs)  # TODO: move this outside, so it doesn't get called multiple times?

        batch_size = len(input_seq)
        initial_state = model.initial_state(batch_size)

        return hk.dynamic_unroll(
            model,
            input_seq,
            initial_state,
            time_major=False,
            return_all_states=return_all_states)


    if beta != 0 or gamma != 0 or delta != 0:
        @jax.jit
        def nll_fn_lstm(params, input_seq, length_mask, beta=0, gamma=0, delta=0):
            """Cross-entropy loss between model-predicted and input behavior."""

            action_probs_seq, stuff = lstm_fn.apply(params, None, input_seq[:, :, input_list])
            # jax.debug.print("action_probs_seq {x}", x=action_probs_seq.shape)
            # jax.debug.print("stuff-1 {x}", x=stuff[-1].shape)
            pmfs, decay_vectors, bias = stuff[-1]

            # make sure there are no 0 probabilities
            action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

            # calculate loss (NLL aka cross-entropy)
            # "sum" and not "mean" so that missed trials don't influence the results
            action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
            action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
            logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions

            # jax.debug.print("contrast_motivation1 {x}", x=pmfs)
            # jax.debug.print("contrast_motivation {x}", x=pmfs.shape)
            pmfs = jnp.diff(pmfs[:, 1:], axis=1)  # compute first derivative
            # jax.debug.print("contrast_motivation {x}", x=pmfs)
            # jax.debug.print("contrast_motivation4 {x}", x=pmfs.shape)
            # jax.debug.print("decay_vectors {x}", x=decay_vectors.shape)
            decay_vectors = jnp.diff(decay_vectors[:, 1:], axis=1)  # compute first derivative
            bias = jnp.diff(bias[:, 1:], axis=1)  # compute first derivative
            # jax.debug.print("decay_vectors {x}", x=decay_vectors.shape)
            # jax.debug.print("len(contrast_motivation) {x}, vs {y}", x=contrast_motivation.shape, y=length_mask.shape)
            # jax.debug.print("normal loss {x}, contrast norm {y}", x=-jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum(), y=beta * jnp.sum(jnp.square(jnp.where(length_mask, contrast_motivation, 0))) / length_mask.sum())
            # jax.debug.print("decay norm {x}", x=jnp.sum(jnp.square(jnp.where(length_mask[..., jnp.newaxis], decay_vectors, 0))))
            nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum() + beta * (jnp.sum(jnp.abs(jnp.where(length_mask[:, 1:, ..., jnp.newaxis, jnp.newaxis], pmfs, 0))) / length_mask.sum()) + \
                                                                                gamma * (jnp.sum(jnp.abs(jnp.where(length_mask[:, 1:, ..., jnp.newaxis], decay_vectors, 0))) / length_mask.sum()) + \
                                                                                delta * (jnp.sum(jnp.abs(jnp.where(length_mask[:, 1:, ..., jnp.newaxis], bias, 0))) / length_mask.sum())

            return nll
    else:
        @jax.jit
        def nll_fn_lstm(params, input_seq, length_mask, beta=0, gamma=0, delta=0):
            """Cross-entropy loss between model-predicted and input behavior."""

            action_probs_seq, stuff = lstm_fn.apply(params, None, input_seq[:, :, input_list])

            # make sure there are no 0 probabilities
            action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

            # calculate loss (NLL aka cross-entropy)
            # "sum" and not "mean" so that missed trials don't influence the results
            action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
            action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
            logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions

            nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

            return nll

    def nll_fn_lstm_with_var(params, input_seq, length_mask):
        """Same as above, but with NLL computation for every session individually as well.
            I wasn't able to get this done via just a bool flag, jax complained
            
            Only works on test data so no beta needed"""

        action_probs_seq, stuff = lstm_fn.apply(params, None, input_seq[:, :, input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (actions were all shifted backwards)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions
        # jax.debug.print("updates {x}", x=beta * sum(jnp.linalg.norm(contrast_motivation)) / length_mask.sum())
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        session_nlls = []
        for i in range(logs.shape[0]):
            session_nlls.append(np.mean(logs[i][length_mask[i]]))
        session_nlls = np.array(session_nlls)

        return nll, session_nlls

    @jax.jit
    def update_func(params, opt_state, input_seq, length_mask):
        """Updates function for the RNN."""

        nll, grads = jax.value_and_grad(nll_fn_lstm)(params, input_seq, length_mask, beta=beta, gamma=gamma, delta=delta)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        # jax.debug.print("updates {x}", x=updates)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, nll


    lstm_fn = hk.transform(_lstm_fn)
    params_lstm = lstm_fn.init(next(rng_seq), input_seq[:, :, input_list])
    opt_state = init_opt(params_lstm)

    init_nll = nll_fn_lstm(params_lstm, input_seq, train_mask, beta=beta, gamma=gamma, delta=delta)
    print(f'Average trialwise probability of initial model to act like mice: {100 * np.exp(-init_nll)}%')

    training_errors = []
    test_errors = []
    steps = []
    params_list = []
    info = {'agent_class': str(agent_class), 'input_list': input_list, 'n_training_steps': n_training_steps,
            'learning_rate': learning_rate, 'weight_decay': weight_decay, 'seed': seed, 'file': file, 'train_batch_size': train_batch_size, 'beta': beta, 'gamma': gamma, 'delta': delta}
    info.update(network_type_to_params[str(agent_class)])  # fill in default values
    info.update(kwargs)  # overwrite with set variables
    info['best_test_nll'] = 100  # placeholder for comparison
    clash_prevention = np.random.randint(100000000) # not so smart

    numpy_rng = np.random.default_rng(seed)  # reuse seed, what could go wrong


    print(file_prefix + "/results/{}_save_{}_intermediate.p".format(network_type_to_name[str(agent_class)], clash_prevention))
    import time
    a = time.time()
    for current_step in range(n_training_steps):

        if train_batch_size:
            train_indices = numpy_rng.choice(input_seq.shape[0], train_batch_size, replace=False)
            params_lstm, opt_state, nll = update_func(params_lstm, opt_state, input_seq[train_indices], train_mask[train_indices])
        else:
            params_lstm, opt_state, nll = update_func(params_lstm, opt_state, input_seq, train_mask)

        if current_step % eval_every == 0:
            steps.append(current_step)
            # nll = nll_fn_lstm(params_lstm, input_seq, train_mask)  # just evaluating on the batch seems not enough?
            nll_test = nll_fn_lstm(params_lstm, input_seq_test, test_mask, beta=0, gamma=0, delta=0)
            training_errors.append(nll)
            test_errors.append(nll_test)
            if nll_test < info['best_test_nll']:
                info['best_net'] = params_lstm
                info['best_test_nll'] = nll_test

        if current_step % 10 == 0:
            print(f'Step {current_step}: test perf. = {100 * np.exp(- info["best_test_nll"])}')
            if save_info:
                all_scalars = pd.DataFrame({'step': steps, 'train_nll': training_errors, 'test_nll': test_errors})
                params_list.append(params_lstm)
                info.update({'all_scalars': all_scalars, 'params_list': params_list})
        if current_step != 0 and (current_step % 5000 == 0 or current_step + 1 == n_training_steps) and save_info:
            pickle.dump(info, open(file_prefix + "/results/{}_save_{}_intermediate.p".format(network_type_to_name[str(agent_class)], clash_prevention), 'wb'))
    print("Loop time {:.4f}".format(time.time() - a))
    
    final_nll = nll_fn_lstm(params_lstm, input_seq, train_mask, beta=beta, gamma=gamma, delta=delta)
    all_scalars = pd.DataFrame({'step': steps, 'train_nll': training_errors, 'test_nll': test_errors})

    print(f'Average trialwise probability of final model to act like mice: {100 * np.exp(-final_nll)}%')

    all_scalars.reset_index(drop=True, inplace=True)

    for col in all_scalars.columns:
        all_scalars[col] = all_scalars[col].astype(float)

    print("Used inputs: {}".format([load_data.input_list[i] for i in input_list]))

    train_nll_lstm = nll_fn_lstm(params_lstm, input_seq, train_mask, beta=beta, gamma=gamma, delta=delta)
    print(f'{agent_class} prediction accuracy on training data: {100 * np.exp(-train_nll_lstm)}%')

    test_nll_lstm, session_nlls = nll_fn_lstm_with_var(params_lstm, input_seq_test, test_mask)
    print(f'{agent_class} prediction accuracy on held-out data: {100 * np.exp(-test_nll_lstm)}%')

    if save_info:
        del info['params_list']  # final results should be slimmer
        info.update({'train_nll_lstm': train_nll_lstm, 'test_nll_lstm': test_nll_lstm, 'all_scalars': all_scalars, 'session_nlls': session_nlls, 'params_lstm': params_lstm})
        pickle.dump(info, open(file_prefix + "/results/{}_save_{}.p".format(network_type_to_name[str(agent_class)], clash_prevention), 'wb'))

    return params_lstm