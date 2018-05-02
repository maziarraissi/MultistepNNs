"""
@author: Maziar Raissi
"""

from Multistep_NN import Multistep_NN

import numpy as np
from scipy.integrate import odeint
    
def main_loop(scheme, M, skip, noise, num_layers, num_neurons):    
    
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
        
    dt = t_star[skip] - t_star[0]
    X_train = X_star[0::skip,:]
    X_train = X_train + noise*X_train.std(0)*np.random.randn(X_train.shape[0], X_train.shape[1])
    
    X_train = np.reshape(X_train, (1,X_train.shape[0],X_train.shape[1]))

    layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [2]]).astype(int).tolist()    
      
    model = Multistep_NN(dt, X_train, layers, M, scheme)
    
    N_Iter = 50000
    model.train(N_Iter)
    
    def learned_f(x,t):
        f = model.predict_f(x[None,:])
        return f.flatten()
    
    learned_X_star = odeint(learned_f, x0, t_star)

    error_x = np.linalg.norm(X_star[:,0] - learned_X_star[:,0], 2)/np.linalg.norm(X_star[:,0], 2)
    error_y = np.linalg.norm(X_star[:,1] - learned_X_star[:,1], 2)/np.linalg.norm(X_star[:,1], 2)
    
    return error_x, error_y        
    

if __name__ == "__main__": 
    
    scheme = ['AB', 'AM', 'BDF']
    M = [1,2,3,4,5]
    
    skip = [1,2,3,4,5]
    noise = [0.0, 0.01, 0.02]
    
    num_layers = [1,2,3]
    num_neurons = [64,128,256]
    
    ##### Table 0
    
    table_0_x = np.zeros((len(scheme), len(M)))
    table_0_y = np.zeros((len(scheme), len(M)))
    
    for i in range(0,len(scheme)):
        for j in range(0,len(M)):
            table_0_x[i,j], table_0_y[i,j] = main_loop(scheme[i], M[j], skip[0], noise[0], num_layers[0], num_neurons[-1])
    
    np.savetxt('./tables/Cubic2D_table_0_scheme_M_x.csv', table_0_x, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')
    np.savetxt('./tables/Cubic2D_table_0_scheme_M_y.csv', table_0_y, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')
    
    
    ##### Table 1
    
    table_1_x = np.zeros((len(skip), len(noise)))
    table_1_y = np.zeros((len(skip), len(noise)))
    
    for i in range(0,len(skip)):
        for j in range(0,len(noise)):
            table_1_x[i,j], table_1_y[i,j] = main_loop(scheme[1], M[0], skip[i], noise[j], num_layers[0], num_neurons[0])
                        
    np.savetxt('./tables/Cubic2D_table_1_dt_noise_x.csv', table_1_x, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')
    np.savetxt('./tables/Cubic2D_table_1_dt_noise_y.csv', table_1_y, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')
    
    
    ##### Table 2
    
    table_2_x = np.zeros((len(num_layers), len(num_neurons)))
    table_2_y = np.zeros((len(num_layers), len(num_neurons)))
    
    for i in range(0,len(num_layers)):
        for j in range(0,len(num_neurons)):
            table_2_x[i,j], table_2_y[i,j] = main_loop(scheme[1], M[0], skip[0], noise[0], num_layers[i], num_neurons[j])
    
    np.savetxt('./tables/Cubic2D_table_2_layers_neurons_x.csv', table_2_x, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')
    np.savetxt('./tables/Cubic2D_table_2_layers_neurons_y.csv', table_2_y, delimiter=' & ', fmt='%.1e', newline=' \\\\\n')