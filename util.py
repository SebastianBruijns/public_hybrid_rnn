import numpy as np

def softmax(activities):
    return np.exp(activities) / np.exp(activities).sum(-1, keepdims=1)

assert np.allclose(softmax(np.arange(3)), np.array([0.09003057, 0.24472847, 0.66524096]))
assert np.allclose(softmax(np.array([[0, 1, 2], [2, 1, 0]])), np.array([[0.09003057, 0.24472847, 0.66524096], [0.66524096, 0.24472847, 0.09003057]]))

def create_exp_filter(decay, length):
  """
    Create an exponential filter.
    Since the network learns a weight with which to multiply the outcome, we remove the normalisation, this will make the decay constant more interpretable.
  """
  weights = np.exp(- decay * np.arange(length))
  return weights[::-1]

def info2name(infos):
    name = ""
    if infos["agent_class"] in ["<class '__main__.Mecha_history'>", "<class 'mecha_history.Mecha_history'>"]:
        if 'contextual_update' in infos and infos['contextual_update']:
            name += "Context-"
        if 'network_memory' in infos and infos['network_memory']:
            name += "Encode"
        if 'network_decay' in infos and infos['network_decay']:
            name += "Decay" if name == "" else "+Decay"
        if 'memory_size' in infos and infos['memory_size'] == 8:
            name += "+Decode"
        if ('network_memory' not in infos or not infos['network_memory']) and ('network_decay' not in infos or not infos['network_decay']):
            name = "Exponential Filter"
        if infos['history_slots'] != 'infinite':
            name += "_{}".format(infos['history_slots'])
    elif infos["agent_class"] in ["<class 'mouse_rnn.LstmAgent'>"]:
        name = "LSTM"
    elif infos["agent_class"] in ["<class 'bi_rnn.BiRNN'>"]:
        name = "BiRNN"
    elif infos["agent_class"] in ["<class 'rMLPs.RNN'>"]:
        name = "RNN"
    elif infos["agent_class"] in ["<class 'mecha_history_plus_lstm.Mecha_history_plust_lstm'>"]:
        name = "LSTM-Mecha"
    else:
        "Unnamed"

    return name