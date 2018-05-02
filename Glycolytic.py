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

if __name__ == "__main__": 
    
    # function that returns dx/dt
    def f(x,t): # x is 3 x 1
        J0 = 2.5
        k1 = 100.0
        k2 = 6.0
        k3 = 16.0
        k4 = 100.0
        k5 = 1.28
        k6 = 12.0
        k = 1.8
        kappa = 13.0
        q = 4
        K1 = 0.52
        psi = 0.1
        N = 1.0
        A = 4.0
        
        f1 = J0 - (k1*x[0]*x[5])/(1 + (x[5]/K1)**q)
        f2 = 2*(k1*x[0]*x[5])/(1 + (x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]
        f3 = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])
        f4 = k3*x[2]*(A-x[5]) - k4*x[3]*x[4] - kappa*(x[3]-x[6])
        f5 = k2*x[1]*(N-x[4]) - k4*x[3]*x[4] - k6*x[1]*x[4]
        f6 = -2*(k1*x[0]*x[5])/(1 + (x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]
        f7 = psi*kappa*(x[3]-x[6]) - k*x[6]
        
        f = np.array([f1,f2,f3,f4,f5,f6,f7])
        return f
        
    # time points
    t_star = np.arange(0,10,0.01)
    
    S1 = np.random.uniform(0.15,1.60,1)
    S2 = np.random.uniform(0.19,2.16,1)
    S3 = np.random.uniform(0.04,0.20,1)
    S4 = np.random.uniform(0.10,0.35,1)
    S5 = np.random.uniform(0.08,0.30,1)
    S6 = np.random.uniform(0.14,2.67,1)
    S7 = np.random.uniform(0.05,0.10,1)
    
    # initial condition
    x0 = np.array([S1,S2,S3,S4,S5,S6,S7]).flatten()
    
    # solve ODE
    X_star = odeint(f, x0, t_star)
    
    noise = 0.00
    
    skip = 1
    dt = t_star[skip] - t_star[0]
    X_train = X_star[0::skip,:]
    X_train = X_train + noise*X_train.std(0)*np.random.randn(X_train.shape[0], X_train.shape[1])
    
    X_train = np.reshape(X_train, (1,X_train.shape[0],X_train.shape[1]))

    layers = [7, 256, 7]
    
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
    
    fig, ax = newfig(1.0, 1.55)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(3, 2)
    gs0.update(top=0.95, bottom=0.35, left=0.1, right=0.95, hspace=0.5, wspace=0.3)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,X_star[:,0],'r-')
    ax.plot(t_star,learned_X_star[:,0],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_1$')
    
    ax = plt.subplot(gs0[0:1, 1:2])
    ax.plot(t_star,X_star[:,1],'r-')
    ax.plot(t_star,learned_X_star[:,1],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_2$')

    ax = plt.subplot(gs0[1:2, 0:1])
    ax.plot(t_star,X_star[:,2],'r-')
    ax.plot(t_star,learned_X_star[:,2],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_3$')
    
    ax = plt.subplot(gs0[1:2, 1:2])
    ax.plot(t_star,X_star[:,3],'r-')
    ax.plot(t_star,learned_X_star[:,3],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_4$')
    
    ax = plt.subplot(gs0[2:3, 0:1])
    ax.plot(t_star,X_star[:,4],'r-')
    ax.plot(t_star,learned_X_star[:,4],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_5$')
    
    ax = plt.subplot(gs0[2:3, 1:2])
    ax.plot(t_star,X_star[:,5],'r-')
    ax.plot(t_star,learned_X_star[:,5],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_6$')

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.25, bottom=0.105, left=0.325, right=0.7, hspace=0.5, wspace=0.3)
    
    ax = plt.subplot(gs1[0:1, 0:2])
    ax.plot(t_star,X_star[:,6],'r-',label='Exact Dynamics')
    ax.plot(t_star,learned_X_star[:,6],'k--',label='Learned Dynamics')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S_7$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    # savefig('./figures/Glycolytic', crop = False)