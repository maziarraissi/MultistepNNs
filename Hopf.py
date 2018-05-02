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
    def f(x,t): # x is 2 x 1
        mu = x[0]
        omega = 1
        A = 1
        
        f1 = 0
        f2 = mu*x[1] - omega*x[2] - A*x[1]*(x[1]**2+x[2]**2)
        f3 = omega*x[1] + mu*x[2] - A*x[2]*(x[1]**2+x[2]**2)

        f = np.array([f1,f2,f3])
        return f
    
    # time points
    t_star = np.arange(0,75,0.1)
    dt = t_star[1] - t_star[0]
    
    # initial condition
    x0 = np.array([[-0.15,2,0],
                   [-0.05,2,0],
                   
                   [.05,.01,0],
                   [.15,.01,0],
                   [.25,.01,0],
                   [.35,.01,0],
                   [.45,.01,0],
                   [.55,.01,0],
                   
                   [.05,2,0],
                   [.15,2,0],
                   [.25,2,0],
                   [.35,2,0],
                   [.45,2,0],
                   [.55,2,0]])
    
    S = x0.shape[0] # number of trajectories
    N = t_star.shape[0] # number of time snapshots
    D = x0.shape[1] # dimension
    
    X_star = np.zeros((S, N, D))
    
    # solve ODE
    for k in range(0,S):
        X_star[k,:,:] = odeint(f, x0[k,:], t_star)
        
    noise = 0
    X_train = X_star
    X_train = X_train + noise*X_train.std(1,keepdims=True)*np.random.randn(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        
    layers = [3, 256, 3]
    
    
    M = 1
    scheme = 'AM'
    model = Multistep_NN(dt, X_train, layers, M, scheme)
    
    N_Iter = 50000
    model.train(N_Iter)
    
    def learned_f(x,t):
        f = model.predict_f(x[None,:])
        return f.flatten()
    
    # initial condition
    learned_x0 = np.array([[-0.15,2,0],
                           [-0.05,2,0],
                           
                           [.05,.01,0],
                           [.15,.01,0],
                           [.25,.01,0],
                           [.35,.01,0],
                           [.45,.01,0],
                           [.55,.01,0],
                           
                           [.05,2,0],
                           [.15,2,0],
                           [.25,2,0],
                           [.35,2,0],
                           [.45,2,0],
                           [.55,2,0],
                           
                           [-0.2,2,0],
                           [-0.1,2,0],
                           
                           [.1,.01,0],
                           [.2,.01,0],
                           [.3,.01,0],
                           [.4,.01,0],
                           [.5,.01,0],
                           [.6,.01,0],
                           
                           [.1,2,0],
                           [.2,2,0],
                           [.3,2,0],
                           [.4,2,0],
                           [.5,2,0],
                           [.6,2,0],
                           
                           [0,2,0],
                           [0,.01,0]])
    
    learned_S = learned_x0.shape[0] # number of trajectories
    learned_N = t_star.shape[0] # number of time snapshots
    learned_D = learned_x0.shape[1] # dimension
    
    learned_X_star = np.zeros((learned_S, learned_N, learned_D))
    
    # solve ODE
    for k in range(0,learned_S):
        learned_X_star[k,:,:] = odeint(learned_f, learned_x0[k,:], t_star)

        
    ####### Plotting ################## 
    fig, ax = newfig(1.0, 0.8)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.1, left=0.0, right=0.90, wspace=0.15)
    
    ax = plt.subplot(gs0[:, 0:1], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    for k in range(0,S):
        colorline3d(ax, X_star[k,:,0], X_star[k,:,1], X_star[k,:,2], cmap = plt.cm.seismic)
    ax.grid(False)
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$y$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    ax = plt.subplot(gs0[:, 1:2], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))    
    for k in range(0,learned_S):
        colorline3d(ax, learned_X_star[k,:,0], learned_X_star[k,:,1], learned_X_star[k,:,2], cmap = plt.cm.seismic)
    ax.grid(False)
    ax.grid(False)
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$y$')
    ax.set_title('Learned Dynamics', fontsize = 10)
    
    # savefig('./figures/Hopf', crop = False)

    