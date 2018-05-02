"""
@author: Maziar Raissi
"""

from Multistep_NN import Multistep_NN

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plotting import newfig, savefig
import matplotlib.gridspec as gridspec

import scipy.io

def colorline3d(ax, x, y, z, cmap):
    N = len(x)
    skip = int(0.01*N);
    for i in range(0,N,skip):
        ax.plot(x[i:i+skip], y[i:i+skip], z[i:i+skip], color=cmap(int(255*i/N)))
  
if __name__ == "__main__": 
    
    
    # Load Data
    data = scipy.io.loadmat('./Cylinder.mat')
    
    X_star = data['X_star']
    
    # time points
    dt = 0.02
    t_star = np.arange(0,X_star.shape[0])*dt
    
    # initial condition
    x0 = X_star[0,:]
        
    skip = 1
    dt = t_star[skip] - t_star[0]
    X_train = X_star[0::skip,:]
    
    X_train = np.reshape(X_train, (1,X_train.shape[0],X_train.shape[1]))

    layers = [3, 256, 3]
    
    M = 1
    scheme = 'AM'
    model = Multistep_NN(dt, X_train, layers, M, scheme)
    
    N_Iter = 50000
    model.train(N_Iter)
    
    def learned_f(x,t):
        f = model.predict_f(x[None,:])
        return f.flatten()
    
    learned_X_star = odeint(learned_f, x0, t_star)
    
    
    ####### Plotting ################## 
    fig, ax = newfig(1.0, 0.8)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.1, left=0.0, right=0.90, wspace=0.15)
    
    ax = plt.subplot(gs0[:, 0:1], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    colorline3d(ax, X_star[:,0], X_star[:,1], X_star[:,2], cmap = plt.cm.jet)
    ax.grid(False)
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    ax.set_zlim([-150,0])
    ax.set_xticks([-200,0,200])
    ax.set_yticks([-150,0,150])
    ax.set_zticks([-150,-75,0])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    ax = plt.subplot(gs0[:, 1:2], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))    
    colorline3d(ax, learned_X_star[:,0], learned_X_star[:,1], learned_X_star[:,2], cmap = plt.cm.jet)
    ax.grid(False)
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    ax.set_zlim([-150,0])
    ax.set_xticks([-200,0,200])
    ax.set_yticks([-150,0,150])
    ax.set_zticks([-150,-75,0])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Learned Dynamics', fontsize = 10)
    
    # savefig('./figures/Cylinder', crop = False)
    
    
    ####### Plotting Vorticity ##################
    
    data = scipy.io.loadmat('./cylinder_vorticity.mat')
    XX = data['XX']
    YY = data['YY']
    WW = data['WW']

    fig, ax = newfig(1.0, 0.65)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.85, bottom=0.2, left=0.25, right=0.8, wspace=0.15)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    h = ax.pcolormesh(XX, YY, WW, cmap='seismic',shading='gouraud', vmin=-5, vmax=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    fig.colorbar(h)

    # savefig('./figures/Cylinder_vorticity', crop = False)