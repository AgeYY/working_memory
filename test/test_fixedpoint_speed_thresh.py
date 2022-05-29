import context
import numpy as np


fixedpoints = np.zeros((1, 3)) # 3d vector
speed_thresh = 1

# store points less than certain speed
hidden_np = np.array([[1, 1, 1], [2, 2, 2], [4, 5, 6]]) # every row is a state vector
hidden_new_np = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]]) # every row is a state vector

speed = np.linalg.norm(hidden_new_np - hidden_np, axis=1)
speed_bool = speed < speed_thresh

fixedpoints = np.vstack((fixedpoints, hidden_new_np[speed_bool]))
fixedpoints = fixedpoints[1:]
print(fixedpoints)
