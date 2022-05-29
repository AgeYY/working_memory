import context
import numpy as np
import matplotlib.pyplot as plt

x_mesh = np.arange(3)
y_mesh = np.arange(3)

x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
x_flat = x_mesh.flatten()
y_flat = y_mesh.flatten()

cords_pca = np.zeros((x_flat.shape[0], 2))
cords_pca[:, 0] = x_flat
cords_pca[:, 1] = y_flat

position = cords_pca
vel = np.copy(position)

edge = int(np.sqrt(position.shape[0]))
x = position[:, 0].reshape(edge, edge)
y = position[:, 1].reshape(edge, edge)
print(position, '\n', x, '\n', y, vel)
