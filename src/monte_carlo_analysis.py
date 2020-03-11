"""
This module implements the anaylsis of the Monte Carlo method to specify the 
suitable number of noise realizations needed 
"""
# preample
import numpy as np
from simulator import NoisyQubitSimulator
import pickle
###############################################################################        
if __name__ == '__main__': 
    # define Paulis 
    initial_states = [np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                     np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                     np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ]
    
    measurement_operators = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    
    # define the simulation parameters
    T      = 1     # total time        
    M      = 4096      # number of discrete time steps
    Omega  = 10       # energy gap of the qubit
    
    # define pulse sequence parameters
    n_x    = 5                                                          # number of pulses in x-direction
    n_y    = 7                                                          # number of pulses in y-direction
    sigma  = 6*T/M                                                      # standard deviation of the Gaussian pulses
    tau_x  = np.array([(k-0.5)/n_x for k in range(1,n_x+1)])*T          # ideal CPMG pulse locations for x-axis
    tau_y  = np.array([(k-0.5)/n_y for k in range(1,n_y+1)])*T          # ideal CPMG pulse locations for y-axis
    A_x    = np.pi*np.ones(tau_x.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for x-axis
    A_y    = np.pi*np.ones(tau_y.shape)/(np.sqrt(2*np.pi*sigma*sigma))  # ideal amplitude of the CPMG pi pulses for y-axis
    
    # define noise random process parameters
    K      = 10000                  # number of realizations of the noise random process
    f      = np.fft.fftfreq(M)*M/T  # vector of discrteized frequencies
    alpha  = 1
    S_Z    = 1*np.array([(1/(fq+1)**alpha)*(fq<50) + (1/40)*(fq>50) + 0.8*np.exp(-((fq-20)**2)/10) for fq in f[f>=0]])  # desired single side band PSD
    alpha  = 1.5
    S_X    = 1*np.array([(1/(fq+1)**alpha)*(fq<20) + (1/96)*(fq>20) + 0.5*np.exp(-((fq-15)**2)/10) for fq in f[f>=0]])  # desired single side band PSD

    # define qubit simulator
    qubit_simulator = NoisyQubitSimulator(T = T , M = M, tau = [tau_x, tau_y, [0]], sigma = sigma, Omega = Omega, K = K, Type = "Gaussian", P_desired = [S_X, None, S_Z])
    
    # apply the pulse sequence
    qubit_simulator.set_pulses([A_x, A_y, [0]])
    
    # simulate measurements
    for idx_state, initial_state in enumerate([initial_states[0] ,initial_states[2], initial_states[4]]):
    
        # initialize an array to store the expectations
        expectations = np.zeros((K,3))
    
        # loop over all realizations
        for idx_U, U in enumerate(qubit_simulator.U):
            # calculate the final state
            final_state = U @ initial_state @ U.conj().T
    
            # calculate the probability of the outcome
            expectations[idx_U, :] = ( np.real( np.trace(final_state @ measurement_operators[0]) ),
                                       np.real( np.trace(final_state @ measurement_operators[1]) ),
                                       np.real( np.trace(final_state @ measurement_operators[2]) ) 
                                     )
        
        f = open("./../datasets/montecarlo_%d.ds"%idx_state, 'wb')
        pickle.dump({"expectations":expectations}, f, -1)
        f.close()
	
	f = open("./../datasets/montecarlo_pulses.ds", 'wb')
	pickle.dump({"h_x":qubit_simulator.h_x, "h_y":qubit_simulator.h_y, "time_range": qubit_simulator.time_range}, f, -1)
	f.close()