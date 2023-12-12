# geneate end of delay neural state from one RNN and then decoded by another RNN. Two RNN's recurrent neurons are aligned by the tuning peaks

'''
from core.net_struct.struct_analyzer import Struct_analyzer

delay_rnn = get the delay rnn's directory, for example, a biasd RNN
decode_rnn = get the decode rnn's directory, for example, a uniform RNN

# load the two RNNs
sa_delay_rnn = Struct_analyzer()
sa_delay_rnn.load(delay_rnn)
_, _, delay_neuron_label, _ = sa_delay_rnn.prepare_label(sigma_rec=0, sigma_x=0, method='rnn_decoder', label_by_mean=False, generate_state_method='delay_ring') # use label_by_mean=False to use tuning peaks not mean; use generate_state_method='delay_ring' to obtain densor state

sa_decode_rnn = Struct_analyzer(decode_rnn)
sa_decode_rnn.load(decode_rnn)
_, _, decode_neuron_label, _ = sa_delay_rnn.prepare_label(sigma_rec=0, sigma_x=0, method='rnn_decoder', label_by_mean=False, generate_state_method='delay_ring') # use label_by_mean=False to use tuning peaks not mean; use generate_state_method='delay_ring' to obtain densor state

def align_arrays(arr1, arr2):
    # align two arrays, so that arr2[indices] would be the closest to arr1
    indices = [np.argmin(np.abs(arr2 - val)) for val in arr1]
    return indices

indices = align_arrays(delay_neuron_label, decode_neuron_label)

rde = rnn_decoder()
rde.load(decode_rnn)
rde.sub.model.bias = rde.sub.model.bias[indices] # align bias
# so do to align recurrent weights and output weights

neural_state = generate delay rnn's state at the end of the delay
out_color = rde.decode(neural_state) # decode by the rde
# compute the error

# The above procedure compute the error for 1 pair of RNN. Please bootstrap RNN pairs multiple times to get the population results.
'''
