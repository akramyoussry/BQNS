"""
This module implements functions for generating the different datasets used for
training the testing of the proposed algorithm.
"""

#preample
import numpy as np
from simulator import NoisyQubitSimulator
from multiprocessing import Pool
from itertools import product
import pickle
###############################################################################
def sim_CPMG_G_MA_1(T, M, Omega, n_x, n_y, power_x, power_y, del_x, del_y, n_max):
    """
    A function to generate one instance of the CPMG Gaussian multi-axis dataset
    
    T      : Total time        
    M      : Number of discrete time steps
    Omega  : Energy gap of the qubit    
    n_x    : CPMG order of the x-axis pulses
    n_y    : CPMG order of the y-axis pulses
    power_x: The random power shift of the x-axis pulses
    power_y: The random power shift of the x-axis pulses
    del_x  : An array of random shifts per pulse in the x-axis
    del_y  : An array of random shifts per pulse in the y-axis
    n_max  : maximum number of pulses in any axis
    """

    # define noise random process parameters
    K      = 1000                  # number of realizations of the noise random process
    f      = np.fft.fftfreq(M)*M/T # vector of discrteized frequencies
    alpha  = 1
    S_Z    = np.array([(1/(fq+1)**alpha)*(fq<50) + (1/40)*(fq>50) + 0.8*np.exp(-((fq-20)**2)/10) for fq in f[f>=0]])  # desired single side band PSD
    alpha  = 1.5
    S_X    = np.array([(1/(fq+1)**alpha)*(fq<20) + (1/96)*(fq>20) + 0.5*np.exp(-((fq-15)**2)/10) for fq in f[f>=0]])  # desired single side band PSD


    # define pulse sequence parameters
    sigma  = 6*T/M                                                      # standard deviation of the Gaussian pulses
    tau_x  = np.array([(k-0.5)/n_x for k in range(1,n_x+1)])*T          # ideal CPMG pulse locations for x-axis
    tau_y  = np.array([(k-0.5)/n_y for k in range(1,n_y+1)])*T          # ideal CPMG pulse locations for y-axis
    A_x    = np.pi*np.ones(tau_x.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for x-axis
    A_y    = np.pi*np.ones(tau_y.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for y-axis
    
    # randomize the pulses
    tau_x = tau_x + del_x * 6 * sigma # add random time shift for x-axis pulses
    tau_y = tau_y + del_y * 6 * sigma # add random time shift for y-axis pulses
    A_x   = A_x * power_x             # randomize ampltudes of the x-axis pulses
    A_y   = A_y * power_y             # randomize ampltudes of the y-axis pulses
    
    # define qubit simulator
    qubit_simulator = NoisyQubitSimulator(T = T , M = M, tau = [tau_x, tau_y, [0]], sigma = sigma, Omega = Omega, K = K, Type = "Gaussian", P_desired = [None, None, None])
    
    # apply the pulse sequence
    qubit_simulator.set_pulses([A_x, A_y, [0]])
    
    # simulate measurements
    ex_targets   = np.reshape(qubit_simulator.measure_all(), (1,18))
    
    # collect input features 
    ex_input_wfm = np.concatenate( (np.reshape(qubit_simulator.h_x, (1,M,1)), np.reshape(qubit_simulator.h_y,(1,M,1)) ), axis=-1)
    
    
    ex_input_params                   = np.zeros( (1, n_max, 4))
    ex_input_params[0,0:tau_x.size,0] = tau_x/T
    ex_input_params[0,0:tau_x.size,1] = 0.5*power_x*np.ones(tau_x.shape)

    ex_input_params[0,0:tau_y.size,2] = tau_y/T
    ex_input_params[0,0:tau_y.size,3] = 0.5*power_y*np.ones(tau_y.shape)

    print("%d,%d"%(n_x,n_y))
    
    return ex_input_params, ex_input_wfm, ex_targets
###############################################################################
def sim_CPMG_G_MA_2(T, M, Omega, n_x, n_y, del_x, del_y, n_max):
    """
    A function to generate one instance of the CPMG Gaussian multi-axis dataset
    
    T      : Total time        
    M      : Number of discrete time steps
    Omega  : Energy gap of the qubit    
    n_x    : CPMG order of the x-axis pulses
    n_y    : CPMG order of the y-axis pulses
    del_x  : An array of random shifts per pulse in the x-axis
    del_y  : An array of random shifts per pulse in the y-axis
    n_max  : maximum number of pulses in any axis
    """

    # define noise random process parameters
    K      = 1000                  # number of realizations of the noise random process
    f      = np.fft.fftfreq(M)*M/T # vector of discrteized frequencies
    alpha  = 1
    S_Z    = np.array([(1/(fq+1)**alpha)*(fq<50) + (1/40)*(fq>50) + 0.8*np.exp(-((fq-20)**2)/10) for fq in f[f>=0]])  # desired single side band PSD
    alpha  = 1.5
    S_X  = np.array([(1/(fq+1)**alpha)*(fq<20) + (1/96)*(fq>20) + 0.5*np.exp(-((fq-15)**2)/10) for fq in f[f>=0]])  # desired single side band PSD


    # define pulse sequence parameters
    sigma  = 6*T/M                                                      # standard deviation of the Gaussian pulses
    tau_x  = np.array([(k-0.5)/n_x for k in range(1,n_x+1)])*T          # ideal CPMG pulse locations for x-axis
    tau_y  = np.array([(k-0.5)/n_y for k in range(1,n_y+1)])*T          # ideal CPMG pulse locations for y-axis
    A_x    = np.pi*np.ones(tau_x.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for x-axis
    A_y    = np.pi*np.ones(tau_y.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for y-axis
    
    # randomize the pulses
    tau_x = tau_x + del_x * 6 * sigma # add random time shift for x-axis pulses
    tau_y = tau_y + del_y * 6 * sigma # add random time shift for y-axis pulses
    
    # define qubit simulator
    qubit_simulator = NoisyQubitSimulator(T = T , M = M, tau = [tau_x, tau_y, [0]], sigma = sigma, Omega = Omega, K = K, Type = "Gaussian", P_desired = [None, None, None])
    
    # apply the pulse sequence
    qubit_simulator.set_pulses([A_x, A_y, [0]])
    
    # simulate measurements
    ex_targets   = np.reshape(qubit_simulator.measure_all(), (1,18))
    
    # collect input features 
    ex_input_wfm = np.concatenate( (np.reshape(qubit_simulator.h_x, (1,M,1)), np.reshape(qubit_simulator.h_y,(1,M,1)) ), axis=-1)
    
    
    ex_input_params                   = np.zeros( (1, n_max, 2))
    ex_input_params[0,0:tau_x.size,0] = tau_x/T
    ex_input_params[0,0:tau_y.size,1] = tau_y/T

    print("%d,%d"%(n_x,n_y))
    
    return ex_input_params, ex_input_wfm, ex_targets

##############################################################################        
if __name__ == '__main__': 

    # define the simulation parameters
    T      = 1         # total time        
    M      = 4096      # number of discrete time steps
    Omega  = 10        # energy gap of the qubit
    
    # generate multi-axis dataset of Gaussian pulses 
    print("Generating multi-axis dataset of Gaussian Pulses")
    
    # initialize the random number generator at some known point for reproducability
    np.random.seed(seed = 30)
    
    n_max       = 7    # maximum number of pulses
    parameters  = [(T, M, Omega, n_x, n_y,  np.random.rand()*2, np.random.rand()*2, (2*np.random.rand(n_x)-1), (2*np.random.rand(n_y)-1), n_max) for n_x,n_y,_ in product(range(1,n_max+1), range(1,n_max+1),range(75))]
    with Pool() as p:
        training_results = p.starmap(sim_CPMG_G_MA_1, parameters)
    
    parameters  = [(T, M, Omega, n_x, n_y,  np.random.rand()*2, np.random.rand()*2, (2*np.random.rand(n_x)-1), (2*np.random.rand(n_y)-1), n_max) for n_x,n_y,_ in product(range(1,n_max+1), range(1,n_max+1),range(25))]
    with Pool() as p:
        testing_results = p.starmap(sim_CPMG_G_MA_1, parameters)

    training_inputs  = [np.concatenate([x[0] for x in training_results], 0), np.concatenate([x[1] for x in training_results], 0)]
    training_targets =  np.concatenate([x[2] for x in training_results], 0)    
    
    testing_inputs   = [np.concatenate([x[0] for x in testing_results], 0), np.concatenate([x[1] for x in testing_results], 0)]
    testing_targets  =  np.concatenate([x[2] for x in testing_results], 0)  
    
    # store the dataset externally in a binary pickle file
    f = open("./../datasets/CPMG_G_XY_%d_nl.ds"%n_max, 'wb')
    pickle.dump({"T":T, "M":M, "Omega":Omega, "training_inputs":training_inputs, "training_targets":training_targets, "testing_inputs":testing_inputs, "testing_targets": testing_targets}, f, -1)
    f.close()
##############################################################################    
    # initialize the random number generator at some known point for reproducability
    np.random.seed(seed = 30)
    
    n_max       = 7    # maximum number of pulses   
    parameters  = [(T, M, Omega, n_x, n_y, (2*np.random.rand(n_x)-1), (2*np.random.rand(n_y)-1), n_max) for n_x,n_y,_ in product(range(1,n_max+1), range(1,n_max+1),range(75))]
    with Pool() as p:
        training_results = p.starmap(sim_CPMG_G_MA_2, parameters)
    
    parameters  = [(T, M, Omega, n_x, n_y, (2*np.random.rand(n_x)-1), (2*np.random.rand(n_y)-1), n_max) for n_x,n_y,_ in product(range(1,n_max+1), range(1,n_max+1),range(25))]
    with Pool() as p:
        testing_results = p.starmap(sim_CPMG_G_MA_2, parameters)

    training_inputs  = [np.concatenate([x[0] for x in training_results], 0), np.concatenate([x[1] for x in training_results], 0)]
    training_targets =  np.concatenate([x[2] for x in training_results], 0)    
    
    testing_inputs   = [np.concatenate([x[0] for x in testing_results], 0), np.concatenate([x[1] for x in testing_results], 0)]
    testing_targets  =  np.concatenate([x[2] for x in testing_results], 0)   
    
    # store the dataset externally in a binary pickle file
    f = open("./../datasets/CPMG_G_XY_pi_%d_nl.ds"%n_max, 'wb')
    pickle.dump({"T":T, "M":M, "Omega":Omega, "training_inputs":training_inputs, "training_targets":training_targets, "testing_inputs":testing_inputs, "testing_targets": testing_targets}, f, -1)
    f.close()
