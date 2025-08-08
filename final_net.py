"""Define explicit history plus MLP networks.
Contrast handling is taken care of by a separate MLP. History is tracked explicitely, and decayed in various ways (let's see what we can do in one class)

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
TODO: Do I want to scale softmaxes with a temp? -> I don't think this is necessary, network and history weight can handle that
TODO: better names
!!!TODO: network decay works on the bunch of 0-vectors with which we initialise, leading to weird results!!!
TRY: Access to all of history, not just the sum
TODO: For extended input, current contrast seems to do better than previous contrast when saving action history vector?
TODO: Only one hidden_size for decay and LSTM
"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from my_module import MirrorInvariantNetwork

Hist_mem = jnp.array

possible_contrasts = jnp.array([jnp.array([1, 0]),
                       jnp.array([0.25, 0]),
                       jnp.array([0.125, 0]),
                       jnp.array([0.0625, 0]),
                       jnp.array([0, 0]),
                       jnp.array([0, 0.0625]),
                       jnp.array([0, 0.125]),
                       jnp.array([0, 0.25]),
                       jnp.array([0, 1])])

default_params = {'n_hiddens': 5,
                  'n_contrast_hiddens': 5,
                  'n_choices': 3,
                  'lstm_contrast_scalar': False,
                  'decay_input': [0, 1, 2],
                  'decay_addon': [],
                  'two_decays': False,
                  'return_all_contrasts': False,
                  }

class Final_net(hk.RNNCore):
  """A bifurcating RNN: "history" processes previous (block) information; "contrast" contrasts."""

  def __init__(self, **kwargs):
    """
      n_hiddens, int - number of hidden units for contrast network
      n_choices, int - number of possible actions, dimension of output
      history_slots, int or 'infinite' - number of history vectors to store, or filter over the entire past if 'infinite'
    """

    super().__init__()

    self._hidden_size = kwargs.get('n_hiddens', default_params['n_hiddens'])
    self.n_contrast_hiddens = kwargs.get('n_contrast_hiddens', default_params['n_contrast_hiddens'])
    self._n_actions = kwargs.get('n_choices', default_params['n_choices'])
    self.lstm_contrast_scalar = kwargs.get('lstm_contrast_scalar', default_params['lstm_contrast_scalar'])
    self.decay_input = kwargs.get('decay_input', default_params['decay_input'])  # what to pass to decay
    self.decay_addon = kwargs.get('decay_addon', default_params['decay_addon'])  # what to pass to decay as addon
    self.two_decays = kwargs.get('two_decays', default_params['two_decays'])
    self.return_all_contrasts = kwargs.get('return_all_contrasts', default_params['return_all_contrasts'])

    self.history_weighting = hk.get_parameter("history_weighting", shape=[1], init=jnp.ones)
    if self.two_decays:
      self.history_weighting_2 = hk.get_parameter("history_weighting_2", shape=[1], init=jnp.ones)

    self.history_initialisation = hk.get_parameter("history_initialisation", shape=[self._n_actions], init=jnp.zeros)
    if self.two_decays:
      self.history_initialisation_2 = hk.get_parameter("history_initialisation_2", shape=[self._n_actions], init=jnp.zeros)
    if self.lstm_contrast_scalar:
      self.lstm_initialisation_1 = hk.get_parameter("lstm_init_1", shape=[self._hidden_size], init=jnp.zeros)
      self.lstm_initialisation_2 = hk.get_parameter("lstm_init_2", shape=[self._hidden_size], init=jnp.zeros)

    # print the params which actually made it
    # print("final net agent")
    # for key in default_params:
    #   print(key + ": " + str(kwargs.get(key, default_params[key])))

  def _state_lstm(self, inputs: jnp.array, prev_state):

    hidden_state, cell_state = prev_state

    forget_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)  # https://dm-haiku.readthedocs.io/en/latest/api.html#linear
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    input_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    candidates = jax.nn.tanh(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    next_cell_state = forget_gate * cell_state + input_gate * candidates

    output_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )
    next_hidden_state = output_gate * jax.nn.tanh(next_cell_state)

    contrast_motivation = hk.Linear(1)(next_hidden_state)

    return contrast_motivation, (next_hidden_state, next_cell_state)
  
  def _decay_network(self, action):

    next_gist = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, self.decay_input], action[:, self.decay_addon])
    if self.two_decays:
        gist_2 = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, self.decay_input], action[:, self.decay_addon])
        next_gist = (next_gist, gist_2)

    return next_gist


  def __call__(self, inputs: jnp.array, total_state: Hist_mem) -> tuple[jnp.array, Hist_mem]:

    if self.lstm_contrast_scalar:
      lstm_state, history_component, (all_contrasts, _, _) = total_state
    else:
      history_component = total_state
    action = inputs[:, :-2]  # shape: (batch_size, n_actions), grab everything before the contrasts, actions and rewards
    contrast = inputs[:, -2:]  # shape: (batch_size, 2), grab the two contrasts

    # update the decays
    decay = self._decay_network(action)

    if self.lstm_contrast_scalar:
      contrast_motivation, lstm_state = self._state_lstm(action, lstm_state)

    # Contrast module: compute contrast influence
    if self.return_all_contrasts:
        next_contrast_gist = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions)(contrast, contrast_motivation if self.lstm_contrast_scalar else None)
        all_contrasts = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions, return_all=True)(self.poss_contrasts, contrast_motivation if self.lstm_contrast_scalar else None)
    else:
        next_contrast_gist = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions)(contrast, contrast_motivation if self.lstm_contrast_scalar else None)

    if not self.two_decays:
      local_action = action[:, 1:4] if len(self.decay_input) == 5 else action[:, :3]
      history_summary = history_component * jax.nn.sigmoid(decay) + local_action
      processed_history = self.history_weighting * history_summary
    else:
      old_hist_1, old_hist_2 = history_component
      decay_1, decay_2 = decay

      # update the 2 histories
      local_action = action[:, 1:4] if len(self.decay_input) == 5 else action[:, :3]
      history_summary_1 = old_hist_1 * jax.nn.sigmoid(decay_1) + local_action
      history_summary_2 = old_hist_2 * jax.nn.sigmoid(decay_2) + local_action

      processed_history = self.history_weighting * history_summary_1 + self.history_weighting_2 * history_summary_2
      history_summary = (history_summary_1, history_summary_2)

    hv_combo = next_contrast_gist + processed_history  # (bs, n_a)
    action_probs = jax.nn.softmax(hv_combo)  # (bs, n_a)

    if self.lstm_contrast_scalar:
      total_state = (lstm_state, history_summary, (all_contrasts, _, _))
    else:
      total_state = history_summary
    return action_probs, total_state

  def initial_state(self, batch_size: Optional[int]) -> Hist_mem:
    self.batch_size = batch_size
    if self.two_decays:
      history_comp = (jnp.tile(self.history_initialisation, (batch_size, 1)), jnp.tile(self.history_initialisation_2, (batch_size, 1)))
    else:
      history_comp = (jnp.tile(self.history_initialisation, (batch_size, 1)))
    if self.return_all_contrasts:
      self.poss_contrasts = jnp.tile(possible_contrasts, (self.batch_size, 1, 1))
    if self.lstm_contrast_scalar:
      return (((jnp.tile(self.lstm_initialisation_1, (batch_size, 1)), jnp.tile(self.lstm_initialisation_2, (batch_size, 1)))),  # lstm state
              history_comp, # history component
              (jnp.zeros((batch_size, 9, 3)) if self.return_all_contrasts else None, None, None)) # raw decay
    else:
      return history_comp

constellations = [{  # network decay is not necessary for contextual update
                'input_list': [0, 1, 2, 5, 3, 4]
            }]

combs = [(True, True, True, False), (True, True, False, True), (True, False, True, True), (False, True, True, True),
         (False, False, False, True), (False, False, True, False), (False, True, False, False), (True, False, False, False),
         (True, True, True, True), (False, False, False, False)]

combs = [(True, True, True, True)]

if __name__ == "__main__":
  import run_rnn_functions
  from itertools import product
  import sys

  constellation = constellations[0]
  sweep_params = list(product([100, 101, 102, 103], [8, 64, 16], [16, 8], [16, 8], [1e-3], [1e-4, 1e-3], combs)) # [0.1, 1, 10, 100]
  seed, train_batch_size, n_hiddens, n_contrast_hiddens, learning_rate, weight_decay, setting = sweep_params[int(sys.argv[1])]

  desired_LSTM_contrast, desired_action_dep_decay, desired_rew_dep_decay, desired_two_histories = setting

  lstm_contrast_scalar = desired_LSTM_contrast
  two_decays = desired_two_histories
  decay_input = [0, 1, 2] if desired_action_dep_decay else [1]
  decay_addon = [3] if desired_rew_dep_decay else []

  return_all_contrasts = False
  beta = 0.
  print("Final net {}".format(constellation))
  print(seed, train_batch_size, n_hiddens, n_contrast_hiddens, learning_rate, weight_decay)

  run_rnn_functions.initialise_and_train(agent_class=Final_net, save_info=False, n_training_steps=200, seed=seed,
                                          train_batch_size=train_batch_size, n_hiddens=n_hiddens, n_contrast_hiddens=n_contrast_hiddens,
                                          learning_rate=learning_rate, weight_decay=weight_decay, lstm_contrast_scalar=lstm_contrast_scalar,
                                          decay_input=decay_input, decay_addon=decay_addon, beta=beta, return_all_contrasts=return_all_contrasts,
                                          two_decays=two_decays, **constellation)
