# plot firing data in the state space
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random

sns.set()
sns.set_style("ticks")

class Manifold_ploter():
    def load_data(self, fir_rate, epoch, target_color):
        '''
        input:
          firing_rate ( array [float] (time_len, ring_centers, rnn_size) )
          target_color: 1 dimensional array. Color is indicated by degree
          epochs (dic) includes:
               - fix: [0, fix_end]
               - stim1: [start, end]
               - interval: [start, end]
               - go_cue: [start, end]
               - response: [start, end]
        '''
        self.fir_rate = np.array(fir_rate).copy()
        self.epoch = epoch.copy()
        self.target_color = target_color.copy()

    def pca_2d_plot_vel(self, start_time=0, end_time=None, ax=None, report_explain_var=False, alpha=0.2, do_pca_fit=True, end_point_size=100):
        pass
        #'''
        #similar with pca_2d_plot, but use velocity to color the trials' trajectories
        #'''
        #plt_colors = self._gen_plt_color() # generate the plt_colors according to the target_color

        #if do_pca_fit:
        #    concate_transform_split = self._pca_fit_transform(2, start_time, end_time, report_explain_var)
        #else:
        #    concate_transform_split = self._pca_transform(2, start_time, end_time, report_explain_var)

        #if ax is None:
        #    fig = plt.figure(figsize=(3, 3))
        #    ax = fig.add_subplot(111)
        #else:
        #    fig = None

        #for i in range(0, len(concate_transform_split)):
        #    colori = plt_colors[i, :] # must be RGBA
        #    X = concate_transform_split[i][:, 0]
        #    Y = concate_transform_split[i][:, 1]
        #    for pi in range(len(X)-1):
        #        r = random.random()
        #        b = random.random()
        #        g = random.random()
        #        color = (r, g, b)
        #        ax.plot(X[pi:pi+2], Y[pi:pi+2], color=color, alpha=alpha)

        #    ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], marker='*', color=colori, alpha=alpha, s=end_point_size) # start points
        #    ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], marker='o', color=colori, alpha=alpha, s=end_point_size) # end points

        #return fig, ax


    def pca_2d_plot(self, start_time=0, end_time=None, ax=None, report_explain_var=False, alpha=0.2, do_pca_fit=True, end_point_size=100):
        '''plot the manifold in 2d
        inputs:
          start_time (float): which time do you wanna start. Note the time is measured by the number of time points, not ms. We recommend you to check self.epoch to find the time interval.
          end_time (float): end time. If None then the final points of the expeiment would be shown
          ax: matplotlib ax. If None we will generate one for you
          report_explain_var (bool): report the variance explained by the first three pcs
          alpha (float): alpha value for transparency
          do_pca_fit (bool): fit data with pca, or just used the current pca transformer.
        outputs:
          ax
        '''
        plt_colors = self._gen_plt_color() # generate the plt_colors according to the target_color

        if do_pca_fit:
            concate_transform_split = self._pca_fit_transform(2, start_time, end_time, report_explain_var)
        else:
            concate_transform_split = self._pca_transform(2, start_time, end_time, report_explain_var)

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
        else:
            fig = None

        for i in range(0, len(concate_transform_split)):
            colori = plt_colors[i, :] # must be RGBA
            X = concate_transform_split[i][:, 0]
            Y = concate_transform_split[i][:, 1]
            ax.plot(X, Y, color=colori, alpha=alpha)

            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], marker='*', color=colori, alpha=alpha, s=end_point_size) # start points
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], marker='o', color=colori, alpha=alpha, s=end_point_size) # end points

        return fig, ax

    def pca_3d_plot(self, start_time=0, end_time=None, ax=None, report_explain_var=False, proj_z_value=-10, alpha_3d=0.7, alpha_proj=0.2, do_pca_fit=True):
        '''plot the manifold in 3d
        inputs:
          start_time (float): which time do you wanna start. Note the time is measured by the number of time points, not ms. We recommend you to check self.epoch to find the time interval.
          end_time (float): end time. If None then the final points of the expeiment would be shown
          ax: matplotlib ax. If None we will generate one for you
          report_explain_var (bool): report the variance explained by the first three pcs
          proj_z_value (float): project the 3d manifold to 2d plane z = proj_z_value
          alpha_3d (float): alpha value for 3d manifold
          alpha_proj (float): alpha value for projection
          do_pca_fit (bool): fit data with pca, or just used the current pca transformer.
        outputs:
          ax
        '''
        plt_colors = self._gen_plt_color() # generate the plt_colors according to the target_color

        if do_pca_fit:
            concate_transform_split = self._pca_fit_transform(3, start_time, end_time, report_explain_var)
        else:
            concate_transform_split = self._pca_transform(3, start_time, end_time, report_explain_var)

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = None

        for i in range(0, len(concate_transform_split)):
            colori = plt_colors[i, :] # must be RGBA
            X = concate_transform_split[i][:, 0]
            Y = concate_transform_split[i][:, 1]
            Z = concate_transform_split[i][:, 2]
            ax.plot(X, Y, Z, color=colori, alpha=alpha_3d)

            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2], marker='*', color=colori, alpha=alpha_3d, s=30) # start points
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2], marker='o', color=colori, alpha=alpha_3d, s=30) # end points

            ## 2d projection
            Z_proj = np.ones(Z.shape) * proj_z_value
            ax.plot(X, Y, Z_proj, color=colori, alpha=alpha_proj)
            ax.scatter(X[0], Y[0], proj_z_value, marker='*', color=colori, alpha=alpha_proj) # start points
            ax.scatter(X[-1], Y[-1], proj_z_value, marker='o', color=colori, alpha=alpha_proj) # start points


        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.zaxis._axinfo['juggled'] = (1,2,0) # check how to set spine in https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot

        ax.grid(False)

        return fig, ax

    def _pca_fit(self, n_components, start_time=0, end_time=None, report_explain_var=False):
        # fit the firing rate data
        if end_time is None:
            end_time = self.epoch['response'][-1] # final time in the expeiment
        fir_rate_epoch = self.fir_rate[start_time:end_time, :, :]
        self.pca = PCA(n_components=n_components)
        concate_fir_rate = fir_rate_epoch.reshape(-1, fir_rate_epoch.shape[-1], order='F') # (time_len * colors, rnn_size)
        self.pca.fit(concate_fir_rate)
        if report_explain_var:
            print('the cumulative variance explained in the first three conponents: ', np.cumsum(self.pca.explained_variance_ratio_))

    def _pca_transform(self, n_components, start_time=0, end_time=None, report_explain_var=False):
        # transform the firing rate to target plotable data
        if end_time is None:
            end_time = self.epoch['response'][-1] # final time in the expeiment
        fir_rate_epoch = self.fir_rate[start_time:end_time, :, :]
        concate_fir_rate = fir_rate_epoch.reshape(-1, fir_rate_epoch.shape[-1], order='F') # (time_len * colors, rnn_size)
        concate_fir_rate_transform = self.pca.transform(concate_fir_rate)
        n_color = len(self.target_color)
        concate_transform_split = np.split(concate_fir_rate_transform, n_color, axis=0)
        return concate_transform_split

    def _pca_fit_transform(self, n_components, start_time=0, end_time=None, report_explain_var=False):
        self._pca_fit(n_components, start_time=start_time, end_time=end_time, report_explain_var=False)
        return self._pca_transform(n_components, start_time=start_time, end_time=end_time, report_explain_var=False)
        ## transform the firing rate to target plotable data
        #if end_time is None:
        #    end_time = self.epoch['response'][-1] # final time in the expeiment
        #fir_rate_epoch = self.fir_rate[start_time:end_time, :, :]
        #self.pca = PCA(n_components=n_components)
        #concate_fir_rate = fir_rate_epoch.reshape(-1, fir_rate_epoch.shape[-1], order='F') # (time_len * colors, rnn_size)
        #self.pca.fit(concate_fir_rate)
        #concate_fir_rate_transform = self.pca.transform(concate_fir_rate)
        #n_color = len(self.target_color)
        #concate_transform_split = np.split(concate_fir_rate_transform, n_color, axis=0)
        #if report_explain_var:
        #    print('the cumulative variance explained in the first three conponents: ', np.cumsum(self.pca.explained_variance_ratio_))

    def _gen_plt_color(self):
        deg_color = Degree_color()
        colors = deg_color.out_color(self.target_color, fmat='RGBA')
        return colors

    def pca_with_fix_point(self, fixedpoints, start_time=0, end_time=None, ax=None, marker='x'):
        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = None

        self.pca_3d_plot(start_time = start_time, end_time = end_time, ax = ax)

        self.pca_array_plot(fixedpoints, ax=ax, marker=marker)
        return fig, ax

    def pca_array_plot(self, batch_array, ax=None, marker='x'):
        '''
        plot several high dimension arrays in the pca space. Before calling this function, _pca_fit must be called in some ways, such as self.pca_3d_plot, or pca_with_fix_point
        batch_array (array [float] (batch_size, rnn_size))
        '''
        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = None

        batch_array_pc = self.pca.transform(batch_array)
        ax.plot(batch_array_pc[:, 0], batch_array_pc[:, 1], batch_array_pc[:, 2], marker)

        return fig, ax
