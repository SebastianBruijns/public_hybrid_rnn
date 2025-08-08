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
                  'fit_history_init': False,
                  'additive_decay': True,
                  'symmetric_contrast_net': False,
                  'lstm_contrast_scalar': False,
                  'omni_scalar_provided': False,
                  'action_encoding': False,
                  'pass_to_enc': False,
                  'pass_to_dec': False,
                  'mirror_enc': False,
                  'mirror_dec': False,
                  'action_flip': True,
                  'reverse_5': False,
                  'return_all_contrasts': False,
                  'two_decays': False,
                  'reduced_non_lstm_info': None
                  }

class Simple_Infer_decay(hk.RNNCore):
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
    self.fit_history_init = kwargs.get('fit_history_init', default_params['fit_history_init'])
    self.additive_decay = kwargs.get('additive_decay', default_params['additive_decay'])
    self.symmetric_contrast_net = kwargs.get('symmetric_contrast_net', default_params['symmetric_contrast_net'])
    self.lstm_contrast_scalar = kwargs.get('lstm_contrast_scalar', default_params['lstm_contrast_scalar'])
    self.omni_scalar_provided = kwargs.get('omni_scalar_provided', default_params['omni_scalar_provided'])
    self.action_encoding = kwargs.get('action_encoding', default_params['action_encoding'])
    self.pass_to_enc, self.pass_to_dec = kwargs.get('pass_to_enc', default_params['pass_to_enc']), kwargs.get('pass_to_dec', default_params['pass_to_dec'])
    self.mirror_enc = kwargs.get('mirror_enc', default_params['mirror_enc'])
    self.mirror_dec = kwargs.get('mirror_dec', default_params['mirror_dec'])
    self.action_flip = kwargs.get('action_flip', default_params['action_flip'])
    self.reverse_5 = kwargs.get('reverse_5', default_params['reverse_5'])
    self.return_all_contrasts = kwargs.get('return_all_contrasts', default_params['return_all_contrasts'])
    self.two_decays = kwargs.get('two_decays', default_params['two_decays'])
    self.reduced_non_lstm_info = kwargs.get('reduced_non_lstm_info', default_params['reduced_non_lstm_info'])

    self.history_weighting = hk.get_parameter("history_weighting", shape=[1], init=jnp.ones)
    if self.two_decays:
      self.history_weighting_2 = hk.get_parameter("history_weighting_2", shape=[1], init=jnp.ones)

    self.history_initialisation = hk.get_parameter("history_initialisation", shape=[self._n_actions], init=jnp.zeros)
    if self.two_decays:
      self.history_initialisation_2 = hk.get_parameter("history_initialisation_2", shape=[self._n_actions], init=jnp.zeros)
    self.decay_initialisation = hk.get_parameter("decay_initialisation", shape=[self._n_actions], init=jnp.zeros)
    if self.lstm_contrast_scalar:
      self.lstm_initialisation_1 = hk.get_parameter("lstm_init_1", shape=[self._hidden_size], init=jnp.zeros)
      self.lstm_initialisation_2 = hk.get_parameter("lstm_init_2", shape=[self._hidden_size], init=jnp.zeros)

    assert not (self.two_decays and self.action_flip), "not implemented"

    # print the params which actually made it
    # print("Infer decay agent")
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

  def _contrast_mlp(self, contrast):

    next_state = jax.nn.tanh(hk.Linear(self.n_contrast_hiddens)(contrast))
    next_gist = hk.Linear(self._n_actions)(next_state)

    return next_gist
  
  def _decay_network(self, action):

    if self.mirror_dec:
      next_gist = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, :3 if not self.reverse_5 else 5], action[:, 3 if not self.reverse_5 else 5:])
      if self.two_decays:
        gist_2 = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, :3 if not self.reverse_5 else 5], action[:, 3 if not self.reverse_5 else 5:])
        next_gist = (next_gist, gist_2)
    else:
      next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(action))
      next_gist = hk.Linear(self._n_actions)(next_state)
      if self.two_decays:
        next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(action))
        next_gist = (next_gist, hk.Linear(self._n_actions)(next_state))

    return next_gist

  def _encoding_network(self, action):

    if self.mirror_enc:
      action_encoding = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, :3 if not self.reverse_5 else 5], action[:, 3 if not self.reverse_5 else 5:])
      if self.two_decays:
        action_encoding_2 = MirrorInvariantNetwork(self._hidden_size, self._n_actions)(action[:, :3 if not self.reverse_5 else 5], action[:, 3 if not self.reverse_5 else 5:])
        action_encoding = (action_encoding, action_encoding_2)
    else:
      next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(action))
      action_encoding = hk.Linear(self._n_actions)(next_state)
      if self.two_decays:
        next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(action))
        action_encoding = (action_encoding, hk.Linear(self._n_actions)(next_state))

    return action_encoding


  def decay_mapping(self, raw_decay, action):
    mask = action[:, 2:3]

    # Create reversed version of the array
    reversed_decay = raw_decay[:, ::-1]

    # Blend between original and reversed based on mask
    new_decay = (1 - mask) * raw_decay + mask * reversed_decay

    return new_decay

  def __call__(self, inputs: jnp.array, total_state: Hist_mem) -> tuple[jnp.array, Hist_mem]:

    if self.lstm_contrast_scalar:
      lstm_state, history_component, (all_contrasts, decay_constant, contrast_bias) = total_state
    else:
      history_memory = total_state
      history_component, updated_raw_decay = history_memory
    action = inputs[:, :-2]  # shape: (batch_size, n_actions), grab everything before the contrasts, actions and rewards
    contrast = inputs[:, -2:]  # shape: (batch_size, 2), grab the two contrasts

    # update the decays
    if self.additive_decay:
      updated_raw_decay = raw_decay + self._decay_network(action)
    else:
        updated_raw_decay = self._decay_network(action[:, :self.reduced_non_lstm_info] if not (self.pass_to_dec != self.omni_scalar_provided) else action[:, :-1])

    if self.lstm_contrast_scalar:
      contrast_motivation, lstm_state = self._state_lstm(action, lstm_state)
    elif self.omni_scalar_provided:
      contrast_motivation = inputs[:, -3:-2]
    # contrast_motivation = jnp.ones_like(contrast_motivation) * 0.8080196062272245
    # jax.debug.print("contrast {x}", x=contrast_motivation)

    # jax.debug.print("orig. decay {x}", x=jax.nn.sigmoid(updated_raw_decay))
    # jax.debug.print("actions {x}", x=action)
    if self.action_flip:
      if self.reverse_5:
        local_decay = self.decay_mapping(updated_raw_decay.copy(), action[:, 1:4])
      else:
        local_decay = self.decay_mapping(updated_raw_decay.copy(), action[:, :3])
    else:
      local_decay = updated_raw_decay
    # jax.debug.print("new decay {x}", x=jax.nn.sigmoid(local_decay))

    # Contrast module: compute contrast influence
    if not self.symmetric_contrast_net:
      next_contrast_gist = self._contrast_mlp(jnp.concatenate([contrast, contrast_motivation], axis=-1) if (self.lstm_contrast_scalar or self.omni_scalar_provided) else contrast)
    else:
      if self.return_all_contrasts:
        next_contrast_gist = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions)(contrast, contrast_motivation if (self.lstm_contrast_scalar or self.omni_scalar_provided) else None)
        all_contrasts = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions, return_all=True)(self.poss_contrasts, contrast_motivation if (self.lstm_contrast_scalar or self.omni_scalar_provided) else None)
      else:
        next_contrast_gist = MirrorInvariantNetwork(self.n_contrast_hiddens, self._n_actions)(contrast, contrast_motivation if (self.lstm_contrast_scalar or self.omni_scalar_provided) else None)

    if not self.two_decays:
      print("TODO: action might look different!")
      history_summary = history_component * jax.nn.sigmoid(local_decay) + (action[:, :3] if not self.action_encoding else self._encoding_network(action[:, :self.reduced_non_lstm_info] if not (self.pass_to_enc != self.omni_scalar_provided) else action[:, :-1]))
      # Combine value and habit
      processed_history = self.history_weighting * history_summary
    else:
      old_hist_1, old_hist_2 = history_component
      decay_1, decay_2 = local_decay

      # update the 2 histories
      if self.reverse_5:
          act_enc_1, act_enc_2 = (action[:, 1:4], action[:, 1:4]) if not self.action_encoding else self._encoding_network(action[:, :self.reduced_non_lstm_info] if not (self.pass_to_enc != self.omni_scalar_provided) else action[:, :-1])
      else:
          act_enc_1, act_enc_2 = (action[:, :3], action[:, :3]) if not self.action_encoding else self._encoding_network(action[:, :self.reduced_non_lstm_info] if not (self.pass_to_enc != self.omni_scalar_provided) else action[:, :-1])
      # jax.debug.print("enc {x} {y}", x=act_enc_1, y=act_enc_2)
      history_summary_1 = old_hist_1 * jax.nn.sigmoid(decay_1) + act_enc_1
      history_summary_2 = old_hist_2 * jax.nn.sigmoid(decay_2) + act_enc_2

      processed_history = self.history_weighting * history_summary_1 + self.history_weighting_2 * history_summary_2
      history_summary = (history_summary_1, history_summary_2)
    # jax.debug.print("processed_history {x}", x=processed_history)
    hv_combo = next_contrast_gist + processed_history  # (bs, n_a)

    action_probs = jax.nn.softmax(hv_combo)  # (bs, n_a)

    if self.lstm_contrast_scalar:
      total_state = (lstm_state, history_summary, (all_contrasts, decay_constant, contrast_bias))
    else:
      total_state = (history_summary, updated_raw_decay)
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
              (jnp.zeros((batch_size, 9, 3)), jnp.zeros((batch_size, 3)), jnp.zeros((batch_size, 1)))) # raw decay
    else:
      return ((history_comp,
              jnp.tile(self.decay_initialisation, (batch_size, 1))))

constellations = [{  # network decay is not necessary for contextual update
                'input_list': [0, 1, 2, 5, 3, 4]
            }]

if __name__ == "__main__":
  import run_rnn_functions
  from itertools import product
  import sys

  constellation = constellations[0]
  sweep_params = list(product([100, 101, 102, 103], [8, 64, 16], [16, 8], [16, 8], [1e-3], [1e-4, 1e-3], [0.])) # [0.1, 1, 10, 100]
  seed, train_batch_size, n_hiddens, n_contrast_hiddens, learning_rate, weight_decay, beta = sweep_params[int(sys.argv[1])]
  additive_decay = False
  symmetric_contrast_net = True
  lstm_contrast_scalar = True
  # constellation['input_list'].insert(-2, extra_input)
  omni_scalar_provided = False
  action_encoding = False
  pass_to_enc, pass_to_dec = False, False
  mirror_enc, mirror_dec = True, True
  reverse_5 = False
  return_all_contrasts = False
  two_decays = True
  action_flip = False
  history_initialisation = True
  print("Simple_Infer_decay agent {}".format(constellation))
  print(seed, train_batch_size, n_hiddens, n_contrast_hiddens, learning_rate, weight_decay)
  reduced_non_lstm_info = 4

  # assert not (constellation['input_list'][0] != 0 and not action_encoding)

  run_rnn_functions.initialise_and_train(agent_class=Simple_Infer_decay, save_info=False, n_training_steps=200, seed=seed,
                                          train_batch_size=train_batch_size, n_hiddens=n_hiddens, n_contrast_hiddens=n_contrast_hiddens,
                                          learning_rate=learning_rate, weight_decay=weight_decay, additive_decay=additive_decay,
                                          symmetric_contrast_net=symmetric_contrast_net, lstm_contrast_scalar=lstm_contrast_scalar,
                                          action_encoding=action_encoding, omni_scalar_provided=omni_scalar_provided,
                                          pass_to_enc=pass_to_enc, pass_to_dec=pass_to_dec, mirror_enc=mirror_enc, mirror_dec=mirror_dec,
                                          beta=beta, return_all_contrasts=return_all_contrasts, reverse_5=reverse_5, two_decays=two_decays,
                                          action_flip=action_flip, history_initialisation=history_initialisation, reduced_non_lstm_info=reduced_non_lstm_info, **constellation)
