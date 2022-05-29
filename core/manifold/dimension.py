# measuring the dimensionality of the manifold
import numpy as np
from .. import run
from sklearn.decomposition import PCA

def pac_var(firing_rate):
    '''
    measuring the variance of explained under various number of principle components
    input:
      firing_rate (array [float] (n, n)): 
    '''
    #for n_components in 
    #pca = PCA(n_components=2)
    pass

def get_firing_rate(rule_name, serial_idx=0, prod_intervals=np.array([1200]), epoch='interval', noise_on=False, ring_centers = np.array([6.]), is_cuda=True):
    '''
    run the model and get the firing rate.
    input:
      rule_name (str): rule name can be color_response_delay_32 or color_response_delay_cones
      serial_idx (str): 'model_i' where i should be replaced by the corresponse directory index
      prod_interval (array [float] (1)): the production interval.
      epoch (str): interval means delay interval
      noise_on (bool): the input noise. Note the hidden noise will always exists unless you set hp['noise_on'] to false
      ring_centers (array [float] (m)): the input color with unit degree.
      is_cuda (bool): run on the cuda or not
    output:
      firing_rate ( array [float] (time_len, ring_centers, rnn_size) )
    '''
    model_dir = '../core/model/'+ rule_name + '/' + str(serial_idx)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)

    runnerObj = run.Runner(model_dir=model_dir, rule_name=rule_name, is_cuda=is_cuda, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, sampled_degree=ring_centers) # will dly_interval also be passed to the run?

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    return firing_rate

