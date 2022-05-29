import context
from core.data_plot import color_reproduction_dly_lib as clib
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

common_color = [45, 135, 225, 315]

target_dire = np.linspace(0, 360, 360)
out_dire0 = np.linspace(-45, 45, 90) + np.random.normal() * 20
out_dire1 = np.linspace(-45, 45, 90) + np.random.normal() * 20
out_dire2 = np.linspace(-45, 45, 90) + np.random.normal() * 20
out_dire3 = np.linspace(-45, 45, 90) + np.random.normal() * 20
out_dire = np.concatenate((out_dire0, out_dire1))
out_dire = np.concatenate((out_dire, out_dire2))
out_dire = np.concatenate((out_dire, out_dire3))

target_common, else_dire = clib.bias_around_common(out_dire, target_dire, common_color)
target_common = target_common.astype(int)

dire_dic = {
    'target_common': target_common,
    'out_dire': out_dire

}

sns.set()
sns.lineplot(data=dire_dic, x='target_common', y='out_dire', ci='sd')
plt.show()
