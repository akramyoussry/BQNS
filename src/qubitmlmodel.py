"""
This module implements the machine learning-based model for the qubit. It has three classes:
    VoConstruction         : This is an internal class for constructing the Vo operators
    HamiltonianConstruction: This is an internal class for constructing Hamiltonians
    QuantumCell            : This is an internal class required for implementing time-ordered evolution
    QuantumEvolution       : This is an internal class to implement time-ordered quantum evolution
    QuantumMeasurement     : This is an internal class to model coupling losses at the output.
    GaussCell              : This is an internal class required for generating time-domain representation of control pulses from signal parameters
    Param_to_Signal_Layer  : This is an internal class that takes the parameterization of a Gaussian waveform and generates the time-domain representation of the contol pulses
    QuantumFidelity        : This is an internal class to evaluate the fidelity between two matrices (Hilbert-Schmidt distance)
    qubitMLmodel           : This is the main class that defines machine learning model for the qubit.  
"""

# Preamble
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model,initializers
import zipfile    
import os
import pickle
from QNS_AS import QNS_AS
###############################################################################
class VoConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes a vector of parameters represneting eigendecompostion and reconstructs a 2x2 Hermitian traceless matrix. 
    """
    
    def __init__(self, O,  **kwargs):
        """
        Class constructor
        
        O: The observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoConstruction, self).__init__(**kwargs)
    
        self.O = tf.constant(O, dtype=tf.complex64)
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # retrieve the two types of parameters from the input: 3 eigenvector parameters and 1 eigenvalue parameter
        U,d = x
        
        # parametrize eigenvector matrix being unitary as in https://en.wikipedia.org/wiki/Unitary_matrix 
        psi   = tf.cast( U[:,0:1], tf.complex64)*1j
        theta = U[:,1:2]
        delta = tf.cast( U[:,2:], tf.complex64)*1j 
        
        # construct the first matrix
        A = tf.linalg.diag(tf.concat([tf.exp(psi), tf.exp(-psi)], -1))
        
        # construct the second matrix
        B1 = tf.expand_dims( tf.concat([tf.cos(theta), tf.sin(-theta)],-1), -1)
        B2 = tf.expand_dims( tf.concat([tf.sin(theta), tf.cos(theta)],-1), -1)
        
        B  = tf.cast( tf.concat([B1,B2],-1), tf.complex64) 
        
        # construct the third matrix
        C = tf.linalg.diag(tf.concat([tf.exp(delta), tf.exp(-delta)], -1))
        
        # multiply all three to get a Unitary (global phase shift is neglected)
        U = tf.matmul(A, tf.matmul(B,C) )
        
        # construct eigenvalue matrix such that it is traceless
        d = tf.concat([d,-d], -1)
        d = tf.cast( tf.linalg.diag(d), tf.complex64)
        
        # construct the Hermitian tracelesss operator from its eigendecompostion
        H = tf.matmul( tf.matmul(U, d), U, adjoint_b=True)    
        
        # expand the observable operator along batch axis
        O = tf.expand_dims(self.O, 0)
        temp_shape = tf.concat( [tf.shape(U)[0:1], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        O = tf.tile(O, temp_shape)
        
        # Construct Vo operator        
        return tf.matmul(O, H)   
##############################################################################
class HamiltonianConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the Hamiltonian parameters as input, and generates the
    Hamiltonain matrix as an output at each time step for each example in the batch
    """
    
    def __init__(self, dynamic_operators, static_operators, **kwargs):
        """
        Class constructor 
        
        dynamic_operators: a list of all operators that have time-varying coefficients
        static_operators : a list of all operators that have constant coefficients
        """
        
        self.dynamic_operators = [tf.constant(op, dtype=tf.complex64) for op in dynamic_operators]
        self.static_operators  = [tf.constant(op, dtype=tf.complex64) for op in static_operators]
           
        # this has to be called for any tensorflow custom layer
        super(HamiltonianConstruction, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        H = []
        # loop over the strengths of all dynamic operators
        for idx_op, op in enumerate(self.dynamic_operators):
            # select the particular strength of the operato
            h = tf.cast(inputs[:,:,idx_op:idx_op+1] ,dtype=tf.complex64)

            # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
            # number of examples and number of time steps of the input
            temp_shape = tf.concat( [tf.shape(inputs)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # repeat the pulse waveform to as 2x2 matrix
            temp_shape = tf.constant(np.array([1,1,2,2],dtype=np.int32))
            h = tf.expand_dims(h,-1)
            h = tf.tile(h, temp_shape)
            
            # Now multiply each operator with its corresponding strength element-wise and add to the list of Hamiltonians
            H.append( tf.multiply(operator, h) )
       
        # loop over the strengths of all static operators
        for op in self.static_operators:          
            # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
            # number of examples and number of time steps of the input
            temp_shape = tf.concat( [tf.shape(inputs)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # Now add to the list of Hamiltonians
            H.append( operator )
        
        # now add all componenents together
        H =  tf.add_n(H)
                            
        return H    
###############################################################################
class QuantumCell(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces one step forward propagator
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        
        # here we define the time-step including the imaginary unit, so we can later use it directly with the expm function
        self.delta_T= tf.constant(delta_T*-1j, dtype=tf.complex64)

        # we must define this parameter for RNN cells
        self.state_size = [1]
        
        # we must call thus function for any tensorflow custom layer
        super(QuantumCell, self).__init__(**kwargs)

    def call(self, inputs, states):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """         
        
        previous_output = states[0] 
        
        # evaluate -i*H*delta_T
        Hamiltonian = inputs * self.delta_T
        
        #evaluate U = expm(-i*H*delta_T)
        U = tf.linalg.expm( Hamiltonian )
        
        # accuamalte U to to the rest of the propagators
        new_output  = tf.matmul(U, previous_output)    
        
        return new_output, [new_output]
###############################################################################
class QuantumEvolution(layers.RNN):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces the time-ordered evolution unitary as output
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
              
        delta_T: time step for each propagator
        """  
        
        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(delta_T)

        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(cell, return_sequences=False,  **kwargs)
      
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        
        # define identity matrix with correct dimensions to be used as initial propagtor 
        dimensions = tf.shape(inputs)
        I          = tf.eye( dimensions[-1], batch_shape=[dimensions[0]], dtype=tf.complex64 )
        
        return super(QuantumEvolution, self).call(inputs, initial_state=[I])         
###############################################################################    
class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input, 
    and generates the measurement outcome probability as output
    """
    
    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor
        
        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """          
        self.initial_state        = tf.constant(initial_state, dtype=tf.complex64)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex64)
    
        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)
            
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x
        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the
        # number of examples of the input
        temp_shape = tf.concat( [tf.shape(Uc)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # add an extra dimension for the initial state and measurement tensors to represent batch
        initial_state        = tf.expand_dims(self.initial_state,0)
        measurement_operator = tf.expand_dims(self.measurement_operator,0)   
        
        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state        = tf.tile(initial_state, temp_shape )
        measurement_operator = tf.tile(measurement_operator, temp_shape)   
        
        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True )
        
        # calculate the probability of the outcome
        expectation = tf.linalg.trace( tf.matmul( tf.matmul( Vo, final_state), measurement_operator) ) 
        
        return tf.squeeze( tf.reshape( tf.math.real(expectation), temp_shape), axis=-1 )
###############################################################################    
class GaussCell(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes signal parameters of one pulse as input, and produces the time domain Gaussian waveform as output
    """
    
    def __init__(self, T, M, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        
        # define and store time range
        self.T          = T
        self.M          = M
        self.time_range = tf.constant( np.reshape( [(0.5*T/M) + (j*T/M) for j in range(M)], (M,1) ) , dtype=tf.float32)

        # we must define this parameter for RNN cells
        self.state_size = [1]
        
        # we must call thus function for any tensorflow custom layer
        super(GaussCell, self).__init__(**kwargs)

    def call(self, inputs, states):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """         
        
        previous_output = states[0] 
        
        
        sigma = 6*self.T/self.M                                       # pulse width        
        tau   = inputs[:,0:1]*self.T                                  # pulse position
        A     = inputs[:,1:2]*(2*np.pi/np.sqrt(2*np.pi*sigma*sigma))  # pulse amplitude

        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the
        # number of examples of the input
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        
        # add an extra for batch
        time_range = tf.expand_dims(self.time_range, 0)
        
        # tile along the other dimenstions
        time_range = tf.tile(time_range, temp_shape)
        
        # repeat the value of the mean,amplitude, and std of the gaussian at ecah time instant for each example
        tau   = tf.reshape( tf.matmul(tau , tf.ones([1,self.M]) ), (tf.shape(time_range)) )
        A     = tf.reshape( tf.matmul(A,    tf.ones([1,self.M]) ), (tf.shape(time_range)) )
        
        gaussian = tf.square(tf.divide(time_range - tau, sigma))
        gaussian = tf.multiply(A, tf.exp(-0.5*gaussian) )
        new_output = previous_output + gaussian 
        
        return new_output, [new_output]
###############################################################################
class Param_to_Signal_Layer(layers.RNN):
    """
    This class defines a custom tensorflow layer that takes the pulse parameters as input, and generates the
    sampled pulses in time domain as output for each example in the batch
    """
    
    def __init__(self, T, M , **kwargs):
        """
        Class constructor.
              
        T: Total time for evolution
        M: Number of discrete time steps
        """  
        
        self.T = T 
        self.M = M
        
        # use the custom-defined GaussCell as base class for the nodes
        cell = GaussCell(T,M)

        # we must call thus function for any tensorflow custom layer
        super(Param_to_Signal_Layer, self).__init__(cell, return_sequences=False,  **kwargs)
      
    def build(self,input_shape):     
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """   
        self.dim = input_shape.as_list()[-1]
        
        # this has to be called for any tensorflow custom layer
        super(Param_to_Signal_Layer,self).build(input_shape)
        
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.M]),dtype=np.int32), tf.constant(np.array([1],dtype=np.int32))],0 )
        
        # define an all-zeros array that we are goinig to use to build the signal with correct dimensions
        h          = tf.zeros( temp_shape, dtype=tf.float32 )
        
        return super(Param_to_Signal_Layer, self).call(inputs, initial_state=[h])         
    
###############################################################################    
class QuantumFidelity(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Vx, Vy,Vz and Uc operators as inputs, and calcualate the fidelity between thm and I,I,I, and some input G repsectively. 
    """
    
    def __init__(self, **kwargs):
        """
        Class constructor
        
        """   
        # we must call thus function for any tensorflow custom layer
        super(QuantumFidelity, self).__init__(**kwargs)
     
    def build(self,input_shape):     
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """ 
        
        self.d = input_shape[0].as_list()[-1]

        # this has to be called for any tensorflow custom layer
        super(QuantumFidelity, self).build(input_shape)   
    
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the two inputs
        U, V =  x
        
        # calculate the fidelity
        F = tf.square( tf.abs( tf.linalg.trace( tf.matmul(U, V, adjoint_a=True) )) )/(self.d**2)

        return F
###############################################################################
class QuantumController(layers.Layer):
    """
    This class defines a custom tensorflow layer that generates a sequence of contro pulses with the proper format such that it can be used by qubitMLmodel class.
    
    """
    def __init__(self, T, M, n_max, num_sig_params, **kwargs):
        """
        class constructor
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        num_sig_params: Number of signal parameters
        """
        # we must call thus function for any tensorflow custom layer
        super(QuantumController, self).__init__(**kwargs)
        
        # store the parameters
        self.n_max           = n_max
        self.T               = T
        self.M               = M
        self.num_sig_params  = num_sig_params
        
        # define custom weights to be used to generate the control sequence
        self.tau = self.add_weight(name = "tau", shape=(1, self.n_max, 1), dtype=tf.float32, trainable=True)
        
        if self.num_sig_params==2:
            self.A   = self.add_weight(name = "A",   shape=(1, self.n_max, 1), dtype=tf.float32, trainable=True)
        else:
            self.A   = self.add_weight(name = "A",   shape=(1, self.n_max, 1), initializer=initializers.Constant(0.5), dtype=tf.float32, trainable=False)
        
        # define the default pulse locations of the CPMG sequence
        self.tau_CPMG = tf.constant( np.reshape( [(k-0.5)/self.n_max for k in range(1,self.n_max+1)], (1,self.n_max,1) )*self.T, dtype=tf.float32 )
    
    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done. 
        
        inputs: this represents the desired gate to implement, which would have dimensions (Batch, d , d) 
        """
        
        sigma = (6*self.T/self.M)
        
        # define pulse locations (and amplitudes if they vary)
        tau = (6 * sigma * tf.tanh(self.tau) + self.tau_CPMG)/self.T
        
        if self.num_sig_params==2:
            A  = tf.sigmoid(self.A)
        else:
            A  = self.A
        
        # combine the parameters into one tensor
        signal_parameters = tf.concat([tau, A] , -1)

        # calculate the time-domain representation
        time_domain_representation = Param_to_Signal_Layer(self.T, self.M)(signal_parameters)
        
        # put them together to form an input to the qubitMLmodel
        if self.num_sig_params==2:
            return [signal_parameters, time_domain_representation] 
        else:
            return [tau, time_domain_representation]
###############################################################################
class qubitMLmodel():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
      
    def __init__(self, delta_T, Omega, mode="Single_Axis", num_sig_params=2):
        """
        Class constructor.

        delta_T         : The time step for the propagators 
        Omega           : The energy gap of the qubit
        mode            : Either "Single_Axis" or "Multi-Axis"
        num_sig_params  : Number of signal parameters of the control pulses
        """
        
        # store the constructor arguments
        self.delta_T               = delta_T
        self.Omega                 = Omega
        self.mode                  = mode
        self.num_sig_params        = num_sig_params
        
        # define lists for stroring the training history
        self.training_history      = []
        self.val_history           = []
        
        # define the initial states which are all the eigenstates of the Paulis
        initial_states = [
                np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) 
                ]
        
        # define the measurements whcih are taken to be all 3 Paulis
        pauli_operators = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
        
        if mode == "Single_Axis":
            # define a tensorflow input layer for the normalized pulse sequence parameters
            pulse_parameters = layers.Input(shape=(None, num_sig_params), name="Pulse_parameters")
        
            # define a second tensorflow input layer for the pulse sequence in time-domain
            pulse_time_domain = layers.Input(shape=(None,1))
        
            # define the custom defined tensorflow layer that constructs the control Hamiltonian from parameters at each time step
            Hamiltonian_ctrl = HamiltonianConstruction(dynamic_operators=[0.5*pauli_operators[0]], static_operators=[0.5*pauli_operators[2]*Omega], name="Hamiltonian")(pulse_time_domain)
        
        else:
            # define a tensorflow input layer for the normalized pulse sequence parameters
            pulse_parameters = layers.Input(shape=(None, 2*num_sig_params), name="Pulse_parameters")
        
            # define a second tensorflow input layer for the pulse sequence in time-domain
            pulse_time_domain = layers.Input(shape=(None,2))
        
            # define the custom defined tensorflow layer that constructs the control Hamiltonian from parameters at each time step
            Hamiltonian_ctrl = HamiltonianConstruction(dynamic_operators=[0.5*pauli_operators[0], 0.5*pauli_operators[1]], static_operators=[0.5*pauli_operators[2]*Omega], name="Hamiltonian")(pulse_time_domain)
        
        # define a first GRU layer that pre-processes the pulse sequence parameters 
        autocorrelation = layers.GRU(10, return_sequences=True)(pulse_parameters)
        
        
        # define a set of 3 different GRUs as a part of generating the parameters of each of the Vo operators  
        autocorrelation = [layers.GRU(60, return_sequences=False)(autocorrelation) for idx in range(3)]

        # define two NNs one for the producing the eigenvector parameters of the Vo operator and another one for the eigenvalues, and repeat for each Vo operator
        Vo = [VoConstruction(O = X, name="V%d"%idx_X)(
                [layers.Dense(3, activation='linear')(autocorrelation[idx_X]),  layers.Dense(1, activation='sigmoid')(autocorrelation[idx_X])]
                )for idx_X,X in enumerate(pauli_operators)]

        # define the custom defined tensorflow layer that constructs the final control propagtor
        Unitary     = QuantumEvolution(delta_T, name="Unitary")(Hamiltonian_ctrl)

        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
                [QuantumMeasurement(rho,X, name="rho%dM%d"%(idx_rho,idx_X))([Vo[idx_X],Unitary]) for idx_X, X in enumerate(pauli_operators)]
                for idx_rho,rho in enumerate(initial_states)]
       
        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, [] ))
        
        # define now the tensorflow model
        self.model    = Model( inputs = [pulse_parameters, pulse_time_domain], outputs = expectations )
        
        # specify the optimizer and loss function for training 
        self.model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
        
        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.model.summary()

     
    def train_model(self, training_x, training_y, epochs):
        """
        This method is for training the model given the training set
        
        training_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        training_y: A numpy array that stores the meeasurement outcomes (number of examples,18).
        epochs    : The number of iterations to do the training     
        """        
        # retreive the batch size from the training dataset
        num_examples  = training_x[0].shape[0]
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.training_history = self.model.fit(training_x, training_y, epochs=epochs, batch_size=num_examples,verbose=2).history["loss"] 
        
    def train_model_val(self, training_x, training_y, val_x, val_y, epochs):
        """
        This method is for training the model given the training set and the validation set
        
        training_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        training_y: A numpy array that stores the meeasurement outcomes (number of examples,18).
        epochs    : The number of iterations to do the training        
        """        
        # retreive the batch size from the training dataset
        num_examples  = training_x[0].shape[0]
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        h  =  self.model.fit(training_x, training_y, epochs=epochs, batch_size=num_examples,verbose=2,validation_data = (val_x, val_y)) 
        self.training_history  = h.history["loss"]
        self.val_history       = h.history["val_loss"]
               
    def predict_measurements(self, testing_x):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """        
        return self.model.predict(testing_x)
    
    def predict_control_unitary(self,testing_x):
        """
        This method is for evaluating the control unitary. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """
        
        # define a new model that connects the input voltage and the GRU output 
        unitary_model = Model(inputs=self.model.input, outputs=self.model.get_layer('Unitary').output)
    
        # evaluate the output of this model
        return unitary_model.predict(testing_x)            
    
    def predict_Vo(self, testing_x):
        """
        This method is for predicting the Vo operators. Usally called after training.
       
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """
          
        # define a new model that connects the inputs to each of the Vo output layers
        Vo_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(V).output for V in ["V0","V1","V2"] ] )
        
        # predict the output of the truncated model. This physically represents <U_I' O U_I>. We still need to multiply by O to get Vo = <O U_I' O U_I>
        Vo = Vo_model.predict(testing_x)
      
        return Vo
           
    def construct_controller(self, T, M, n_max):
        """
        This method is to build a generic controller for the qubit
        """
        if (self.mode != "Single_Axis" or self.num_sig_params!= 2):
			raise Exception("The current implementation only supports single-axis mode, with 2 parameters for the control signal")
		
        # define input layer for each of the targets [Vx,Vy,Vz,Uc]   
        target_R       = [ layers.Input( shape=(2,2), name="Target_%d_real"%idx ) for idx in range(4) ] 
        target_I       = [ layers.Input( shape=(2,2), name="Target_%d_imag"%idx ) for idx in range(4) ] 

        # construct the complex matrix
        target_complex = [ layers.Lambda(lambda x: tf.cast(x[0], tf.complex64) + 1j*tf.cast(x[1], tf.complex64), name="Target_%d_complex"%idx )([ target_R[idx], target_I[idx] ]) for idx in range(4)]
        
        # define a custom quantum controller layer to obtain the pulse sequence
        if self.mode=="Single_Axis":
            pulse_sequence = QuantumController(T, M, n_max, self.num_sig_params, name="Control_Pulses")(target_complex[3])
        else:
            pulse_sequence = layers.Concatenate(name="Control_Pulses", axis=-1)([QuantumController(T, M, n_max, self.num_sig_params)(target_complex[3]), QuantumController(T, M, n_max)(target_complex[3])])
        
        # extract the part of the pre-trained qubit model & prevent it from training again
        qubit_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(V).output for V in ["V0","V1","V2", "Unitary"] ] , name='qubit_model')
        
        for layer in qubit_model.layers:
            layer.trainable = False
        
        # apply the control sequence and obtain the Vo and Uc
        controlled_complex = qubit_model(pulse_sequence)
        
        # define a custom fidelity layer for each target
        fidelity      = [layers.Reshape((1,), name="Fid_%d"%idx)(
                            QuantumFidelity()([ target_complex[idx], controlled_complex[idx] ])
                            )for idx in range(4)]
        
        # define a tensorflow model for the overall controller structure
        self.controller_model = Model(inputs = target_R + target_I, outputs=fidelity)
        
        # specify the optimizer and loss function for training, with the same weight for all targets
        self.controller_model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'], loss_weights=[0.3, 0.3, 0.3, 0.1])

        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.controller_model.summary()
    
    
    def train_controller(self, target_U, epochs):
        """
        This method is for training the controller to obtain some target unitary
        
        target_G: The target quantum gate to be designed
        epochs  : The number of training iterations
        """
        if (self.mode != "Single_Axis" or self.num_sig_params!= 2):
			raise Exception("The current implementation only supports single-axis mode, with 2 parameters for the control signal")
		      
        target_G = np.reshape( target_U, (1,2,2) )
        
        # define identity matrix to be used as targets for Vo
        I = np.reshape( np.eye(2), (1,2,2) )
        
        training_inputs  = [np.real(I), np.real(I), np.real(I), np.real(target_G), np.imag(I), np.imag(I), np.imag(I), np.imag(target_G)]
        training_targets = [np.ones((1)) for _ in range(4)]

        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.controller_training_history = self.controller_model.fit(training_inputs, training_targets, epochs=epochs, batch_size=1,verbose=2).history 
    
        # retrieve the control sequence
        control_pulses_model     = Model(inputs = self.controller_model.input, outputs = self.controller_model.get_layer("Control_Pulses").output)
        predicted_control_pulses = control_pulses_model.predict(training_inputs)  
        return predicted_control_pulses
    
    
    def estimate_spectrum(self, T, M, n_max):
        """
        This method is to estimate the power spectral density using Alvarez-Suter method
        
        n_max: number of CPMG sequences        
        """
        if (self.mode != "Single_Axis" or self.num_sig_params!= 2):
			raise Exception("The current implementation only supports single-axis mode, with 2 parameters for the control signal")
		
        # define a model to generate the time domain representation of the CPMG sequences
        signal_parameters_input = layers.Input((n_max, 2))     
        time_wfm                = Param_to_Signal_Layer(T, M)(signal_parameters_input)
        wfmModel                = Model(inputs = signal_parameters_input, outputs= time_wfm)
        
        signal_parameters       = np.zeros((n_max, n_max, 2))
        signal_time_domain      = np.zeros((n_max, M, 1))

        for n in range(1,n_max):
            # define ideal CPMG pulse sequence position
            signal_parameters[n,0:n,0:1] = np.reshape( [(k-0.5)/n for k in range(1,n+1)], (1,n,1) )
            # define idal cPMS sequence powers
            signal_parameters[n,0:n,1:2] = 0.5*np.ones( (1,n,1) )
        
        # add the free evolution measurement
        signal_parameters[0,0:1,0:1] = 0.5
        
        signal_time_domain = wfmModel.predict(signal_parameters)
        
        # calculate the predicted coherence using the CPMG sequences 
        Vx         = self.predict_Vo([signal_parameters, signal_time_domain])[0]
        coherences = np.abs( np.reshape( [np.trace( V @ np.array([[0.5, 0.5], [0.5, 0.5]]) ) for V in Vx], (n_max,1) ) ) #(note: Uc rho Uc' X = rho)
        
        # calculate the swithcing waveform
        time_range      = np.resize([(0.5*T/M) + (j*T/M) for j in range(M)], (1,M,1)) 
        switching_wfm   = np.zeros( (n_max,M,1) )   
        for n in range(n_max):   
            t                  = np.array([0] + [(k-0.5)/n for k in range(1,n+1)] + [1])*T
            switching_wfm[n,:] = np.array([sum([((-1.0)**k)*(tau>=t[k])*(tau<t[k+1]) for k in range(n+1)]) for tau in time_range])                

            
        AS_model               = QNS_AS(M, M/T)
        AS_model.train_model(switching_wfm, coherences, 1000)
        
        return signal_parameters, signal_time_domain,switching_wfm, coherences, AS_model.predict_PSD() 
    
    def save_model(self, filename):
        """
        This method is to export the model to an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """
        
        # first save the ml model
        self.model.save_weights(filename+"_model.h5")
        
        # second, save all other variables
        data = {'training_history':self.training_history, 
                'val_history'     :self.val_history,
                }
        f = open(filename+"_class.h5", 'wb')
        pickle.dump(data, f, -1)
        f.close()
	
        # zip everything into one zip file
        f = zipfile.ZipFile("./../datasets/"+filename+".mlmodel", mode='w')
        f.write(filename+"_model.h5")
        f.write(filename+"_class.h5")
        f.close()
        
        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_class.h5")

    def load_model(self, filename):
        """
        This method is to import the models from an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """       
        #unzip the zipfile
        f = zipfile.ZipFile("./../datasets/"+filename+".mlmodel", mode='r')
        f.extractall()
        f.close()
        
        # first load the ml model
        self.model.load_weights(filename+"_model.h5")

                
        # second, load all other variables
        f = open(filename+"_class.h5", 'rb')
        data = pickle.load(f)
        f.close()          
        self.training_history  = data['training_history']
        self.val_history       = data['val_history']

        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_class.h5")
###############################################################################
