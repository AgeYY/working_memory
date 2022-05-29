import numpy as np

a = np.zeros((300, 20))
a_split = np.split(a, 10, axis=0)
print(len(a_split), a_split[0].shape)
