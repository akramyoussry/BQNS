"""
This module implements a simulator for a noisy qubit
"""

# preample
import numpy as np
from functools import reduce
from itertools import product
###############################################################################
class NoisyQubitSimulator(object):
    """
    Class for simulating a noisy spin qubit 
    """

    def __init__(self, T, M, tau, sigma, Omega, K, Type="Gaussian", P_desired=[None,None,None]):
        """
        Class constructor
        
        T         : The total evolution time
        M         : The number of discrete time steps 
        tau       : A list of the lists of centres of the pulses along each direction, put [-1] for no pulse (t is in [0,T])
        sigma     : The standard deviation of the gaussian pulses along each direction/ or the pulse width in case of square pulses
        Omega     : The energy gap of the qubit 
        K         : The number of realizations for the monte carlo simulation
        Type      : Pulse shape which is either "Square" or "Gaussian"(default)
        P_desired : A list of PSD of noise along each direction, put None for a noiseless direction (default is noiseless)
        """
        
        # store the simulation parameters
        self.T         = T
        self.M         = M
        self.tau       = tau
        self.sigma     = sigma
        self.Omega     = Omega
        self.K         = K
        self.Type      = Type
        self.P_desired = P_desired
        
        # initialize the pulse time-domain waveform arrays  
        self.h_x = np.zeros((1,self.M))    
        self.h_y = np.zeros((1,self.M))
        self.h_z = np.zeros((1,self.M))
        
        # construct the time vector
        self.delta_t    = T/M                                                        # time step
        self.time_range =  [(0.5*self.delta_t) + (j*self.delta_t) for j in range(M)] # list of time steps


        # construct the matrix representing the Gaussian bases for each directions        
        if self.Type=="Gaussian":
            self.theta_x = np.array( [[np.exp(-0.5*((t-tau)/self.sigma)**2) for tau in self.tau[0]] for t in self.time_range] )
            self.theta_y = np.array( [[np.exp(-0.5*((t-tau)/self.sigma)**2) for tau in self.tau[1]] for t in self.time_range] )
            self.theta_z = np.array( [[np.exp(-0.5*((t-tau)/self.sigma)**2) for tau in self.tau[2]] for t in self.time_range] )
        
        
        # define the Pauli matrices
        self.sigma_x = np.array([[0.,1.],[1.,0.]])
        self.sigma_y = np.array([[0.,-1j],[1j,0.]])
        self.sigma_z = np.array([[1.,0.],[0.,-1.]])
        
        
    def set_pulses(self, alpha):
        """
        This method to construct the evolution matrix for all noise realizations given the pulses amplitudes
        
        alpha:  A list of the lists of amplitudes of the pulses along each direction
        """
        
        # unpack the amplitude vector for each direction
        self.alpha_x,self.alpha_y,self.alpha_z = alpha
        
        # construct the waveforms
        if self.Type == "Gaussian":           
            self.h_x = (self.theta_x @ self.alpha_x)
            self.h_y = (self.theta_y @ self.alpha_y)
            self.h_z = (self.theta_z @ self.alpha_z)           
        else:
            pwidth = self.sigma
            self.h_x = sum( [np.array([(t>(tau-0.5*pwidth))*(t<(tau+0.5*pwidth))*A for t in self.time_range]) for tau, A in zip(self.tau[0],self.alpha_x)] )
            self.h_y = sum( [np.array([(t>(tau-0.5*pwidth))*(t<(tau+0.5*pwidth))*A for t in self.time_range]) for tau, A in zip(self.tau[1],self.alpha_y)] )
            self.h_z = sum( [np.array([(t>(tau-0.5*pwidth))*(t<(tau+0.5*pwidth))*A for t in self.time_range]) for tau, A in zip(self.tau[2],self.alpha_z)] )
        
        # generate the noise realizations
        self.generate_arbitrary_noise(self.P_desired)
        
        # construct a list of the Hamiltonians for each noise realization
        self.set_hamiltonians()
        
        # construct the unitary matrix for each realization
        self.evolve()
    
    
    def generate_arbitrary_noise(self,P_desired):
        """
        generate random noise according to some desired power spectral density according to the algorithm here:
        https://stackoverflow.com/questions/25787040/synthesize-psd-in-matlab
        
        P_desired: a list of arrays representing desired PSD [single side band representation] along x,y,z
        """
        
        Ts = self.delta_t  # sampling time (1/sampling frequency)
        N  = self.M        # number of required samples
        
        if not P_desired[0] is None:
            # define a list to store the different noise realizations
            self.beta_x  = []
            
            # generate different realizations
            for _ in range(self.K):
                #1) add random phase to the properly normalized PSD
                P_temp = np.sqrt(P_desired[0]*N/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,N//2))
            
                #2) add the symmetric part of the spectrum
                P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )
            
                #3) take the inverse Fourier transform
                x      = np.real(np.fft.ifft(P_temp))
                
                # store
                self.beta_x.append(np.reshape(x,self.h_x.shape))
        else:
            # no noise in this direction
            self.beta_x = [np.zeros(self.h_x.shape) for k in range(self.K)]
        
        if not P_desired[1] is None:        
            # define a list to store the different noise realizations
            self.beta_y  = []    
            # generate different realizations
            for _ in range(self.K):
                #1) add random phase to the properly normalized PSD
                P_temp = np.sqrt(P_desired[1]*N/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,N//2))
            
                #2) add the symmetric part of the spectrum
                P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )
            
                #3) take the inverse Fourier transform
                x      = np.real(np.fft.ifft(P_temp))
                
                # store
                self.beta_y.append(np.reshape(x,self.h_y.shape))
        else:
            # no noise in this direction
             self.beta_y = [np.zeros(self.h_y.shape) for k in range(self.K)]
        
        if not P_desired[2] is None:
            # define a list to store the different noise realizations
            self.beta_z  = []   
            # generate different realizations
            for _ in range(self.K):
                #1) add random phase to the properly normalized PSD
                P_temp = np.sqrt(P_desired[2]*N/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,N//2))
            
                #2) add the symmetric part of the spectrum
                P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )
            
                #3) take the inverse Fourier transform
                x      = np.real(np.fft.ifft(P_temp))
                
                # store
                self.beta_z.append(np.reshape(x,self.h_z.shape)) 
        else:
            # no noise in this direction
            self.beta_z = [np.zeros(self.h_z.shape) for k in range(self.K)]               
              
    def set_hamiltonians(self):
        """
        This method is to construct a list of Hamiltonians to calculate the propagators
        """
                
        # construct and store the Hamitlonian at each time step for all noise realizations
        self.Hamiltonians = [ [0.5 * self.sigma_z * (self.Omega + b_z + h_z) + 0.5 * self.sigma_x * (h_x + b_x) + 0.5 * self.sigma_y * (h_y + b_y) for b_x, b_y, b_z, h_x, h_y,h_z in zip(beta_x, beta_y, beta_z, self.h_x,self.h_y,self.h_z)] for beta_x,beta_y,beta_z in zip(self.beta_x,self.beta_y,self.beta_z)]

    
    def evolve(self):
        """
        This method is to calculate the final unitary
        """
        
        # define a lambda function for calculating the propagators
        evolve = lambda U,U_j: U_j @ U
      
        # calculate and accumalate all propagators till the final one, and repeat over all realizations
        self.U = [reduce(evolve, [self.expm2(self.delta_t*H) for H in Hamiltonian]) for Hamiltonian in self.Hamiltonians]
        
    def measure(self, initial_state, measurement_operator):
        """
        This method is to perfrom measurements on the final state.
        
        initial_state       : The density matrix of the initial state
        measurement_operator: Measurement Operator
        """
        # initialize an empty list to store the expectation for each realization
        expectation = []
        
        # loop over all realizations
        for U in self.U:
            # calculate the final state
            final_state = U @ initial_state @ U.conj().T
        
            # calculate the probability of the outcome
            expectation.append( np.real( np.trace(final_state @ measurement_operator) ) )

        return np.average(expectation)

    def measure_all(self):
        """
        This method to simulate the full tomogrpahic set of measurements with all initial states and all measurement operators
        """
        
        # define a list of initial states corresponding to the up/down eignestates of each of the Pauli measurement operators
        initial_states = [np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                         np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                         np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ]
        
        # define the list of measurement operators
        measurement_operators = [self.sigma_x, self.sigma_y, self.sigma_z]
        
        # calculate each measurement 
        expectations = [self.measure(rho,X) for rho,X in product(initial_states, measurement_operators) ]
        
        return np.array(expectations)
    
    def measure_one_shot(self, initial_state, measurement_projector):
        """
        This method simulates one shot mesaurements returns a list of +1/-1 correspnding to each measurement
        """
        
        # simulate the coin flip with outocomes +1/-1, repeated for each noise realization
        return [2*int( np.random.rand() > np.real(np.trace( U @ initial_state @ U.conj().T @ measurement_projector ) ) )-1 for U in self.U]
    
    def expm2(self, H):
        """
        This is an internal method to caclulate the matrix exponential more efficiently using Euler formula. Works only for qubits.
        """
        
        # parameterize the Hamiltonian in terms of the three Pauli basis
        a_vector = [np.real(H[0,1]), np.imag(H[0,1]), H[0,0]]
        
        # calculate the norm of the Pauli vector
        a = np.sqrt(a_vector[0]**2 + a_vector[1]**2 + a_vector[2]**2 )
        
        if a==0:
            return np.array([[1.,0.],[0.,1.]]) # Identity
        else:
            # use Euler's formula to calculate e^(-i a \hat{n} \cdot \sigma) = I cos a - i (\hat{n}\cdot\sigma) sin a
            return np.cos(a)*np.array([[1.,0.],[0.,1.]]) - 1j*H*np.sin(a)/a