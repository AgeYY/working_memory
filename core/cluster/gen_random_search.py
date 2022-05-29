import json
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

def random_combin(para_space, n):
    '''
    output the randomized combination of parameters
    input:
        n (int): the number of combinations
    '''
    combination = {}
    for key in para_space:
        combination[key] = np.random.choice(para_space[key], size=n)
    return combination

def gen_save_combin(out_path=None, n=None):
    '''
    generate random combinations to file out_path. If n is None, the output would be an empty file. This will lead to the model train on the default configuration in default.py.
    '''
    if out_path is None:
        out_path = '../para_combin.csv'
    with open('../para_space.json') as f:
        para_space = json.load(f)

    if n is None:
        comb_df = pd.DataFrame({})
        comb_df.to_csv(out_path)
    else:
        combination = random_combin(para_space, n)
        comb_df = pd.DataFrame(combination)
        comb_df.to_csv(out_path)

if __name__ == "__main__":
    try:
        gen_save_combin(n=int(sys.argv[1]))
    except:
        gen_save_combin(n=1)
