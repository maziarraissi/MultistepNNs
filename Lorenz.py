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

def colorline3d(ax, x, y, z, cmap):
    N = len(x)
    skip = int(0.01*N)
    for i in range(0,N,skip):
        ax.plot(x[i:i+skip+1], y[i:i+skip+1], z[i:i+skip+1], color=cmap(int(255*i/N)))

if __name__ == "__main__": 
    
    # function that returns dx/dt
    def f(x,t): # x is 3 x 1
        sigma = 10.0
        beta = 8.0/3.0
        rho = 28.0
        
        f1 = sigma*(x[1]-x[0])
        f2 = x[0]*(rho-x[2])-x[1]
        f3 = x[0]*x[1]-beta*x[2]
        f = np.array([f1,f2,f3])
        return f
        
    # time points
    t_star = np.arange(0,25,0.01)
    
    # initial condition
    x0 = np.array([-8.0, 7.0, 27])
    
    # solve ODE
    X_star = odeint(f, x0, t_star)
    
    noise = 0.00
    
    skip = 1
    dt = t_star[skip] - t_star[0]
    X_train = X_star[0::skip,:]
    X_train = X_train + noise*X_train.std(0)*np.random.randn(X_train.shape[0], X_train.shape[1])
    
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
    colorline3d(ax, X_star[:,0], X_star[:,1], X_star[:,2], cmap = plt.cm.ocean)
    ax.grid(False)
    ax.set_xlim([-20,20])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,50])
    ax.set_xticks([-20,0,20])
    ax.set_yticks([-40,0,40])
    ax.set_zticks([0,25,50])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    ax = plt.subplot(gs0[:, 1:2], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))    
    colorline3d(ax, learned_X_star[:,0], learned_X_star[:,1], learned_X_star[:,2], cmap = plt.cm.ocean)
    ax.grid(False)
    ax.set_xlim([-20,20])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,50])
    ax.set_xticks([-20,0,20])
    ax.set_yticks([-40,0,40])
    ax.set_zticks([0,25,50])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Learned Dynamics', fontsize = 10)
    
    # savefig('./figures/Lorenz', crop = False)
    
    ####### Plotting ##################
    
    fig, ax = newfig(1.0, 1.5)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(3, 1)
    gs0.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.5)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,X_star[:,0],'r-')
    ax.plot(t_star,learned_X_star[:,0],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    
    ax = plt.subplot(gs0[1:2, 0:1])
    ax.plot(t_star,X_star[:,1],'r-')
    ax.plot(t_star,learned_X_star[:,1],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y$')
    
    ax = plt.subplot(gs0[2:3, 0:1])
    ax.plot(t_star,X_star[:,2],'r-',label='Exact Dynamics')
    ax.plot(t_star,learned_X_star[:,2],'k--',label='Learned Dynamics')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$z$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    # savefig('./figures/Lorenz_Traj', crop = False)