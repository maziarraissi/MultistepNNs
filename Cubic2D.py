"""
@author: Maziar Raissi
"""

from Multistep_NN import Multistep_NN

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from plotting import newfig, savefig
import matplotlib.gridspec as gridspec

if __name__ == "__main__": 
    
    # function that returns dx/dt
    def f(x,t): # x is 2 x 1
        A = np.array([[-.1,2], [-2,-.1]]) # 2 x 2
        f = np.matmul(A,x[:,None]**3) # 2 x 1
        return f.flatten()
    
    # time points
    t_star = np.arange(0,25,0.01)
    
    # initial condition
    x0 = np.array([2,0])
    
    # solve ODE
    X_star = odeint(f, x0, t_star)
    
    noise = 0.00
    
    skip = 1
    dt = t_star[skip] - t_star[0]
    X_train = X_star[0::skip,:]
    X_train = X_train + noise*X_train.std(0)*np.random.randn(X_train.shape[0], X_train.shape[1])
    
    X_train = np.reshape(X_train, (1,X_train.shape[0],X_train.shape[1]))
    
    layers = [2, 256, 2]
    
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
    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.85, bottom=0.25, left=0.1, right=0.95, wspace=0.3)
    
    ax = plt.subplot(gs0[:, 0:1])
    ax.plot(t_star,X_star[:,0],'r',label='$x$')
    ax.plot(t_star,X_star[:,1],'b',label='$y$')
    ax.plot(t_star,learned_X_star[:,0],'k--',label='learned model')
    ax.plot(t_star,learned_X_star[:,1],'k--')    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x, y$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.25), ncol=3, frameon=False)
    ax.set_title('Trajectories', fontsize = 10)
    
    ax = plt.subplot(gs0[:, 1:2])
    ax.plot(X_star[:,0],X_star[:,1], 'm', label='$(x,y)$')
    ax.plot(learned_X_star[:,0],learned_X_star[:,1],'k--')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.25), ncol=1, frameon=False)
    ax.set_title('Phase Portrait', fontsize = 10)

    # savefig('./figures/Cubic2D')