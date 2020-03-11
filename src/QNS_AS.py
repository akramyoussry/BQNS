"""
This module implements the Alvarez-Suter algorithm for quantum noise spectroscopy.

"""
# preample
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,initializers,Model,constraints
from scipy.linalg import dft
###############################################################################
class coherence_Layer(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the switching function y(tau) as input, and evaluates the
    coherence for each example in the batch
    """
    
    def __init__(self, fs, **kwargs):
        """
        Class constructor 
        """
        
        self.fs = fs 
        # this has to be called for any tensorflow custom layer
        super(coherence_Layer, self).__init__(**kwargs)
    
    
    def build(self,input_shape):
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """    
        # get the size of the input
        self.N = input_shape.as_list()[1]
        
        # construct the DFT matrix
        self.F = dft(self.N)/self.fs
                
        # construct the trapezoidal rule matrix
        self.D        = np.eye(self.N//2)
        self.D[0,0]   = 0.5
        self.D[-1,-1] = 0.5
        
        # define the trainable parameters representing the double side band PSD of noise
        self.S = self.add_weight(name="S", shape=tf.TensorShape((self.N//2,1)), initializer=initializers.Ones(), constraint = constraints.NonNeg(), trainable=True)
               
        # this has to be called for any tensorflow custom layer
        super(coherence_Layer, self).build(input_shape)

    
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # calculate the FFT of the switching function
        x_tilde = tf.cast( x , tf.complex64)
        
        # retrieve the DFT matrix
        F = tf.constant(self.F, dtype=tf.complex64)
        
        # add extra dimension for batch
        F = tf.expand_dims(F, 0)
        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the
        # number of example of the input
        temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # repeat the matrix along the batch dimension
        F = tf.tile( F, temp_shape )
        
        # apply the complex operator to convert lower traingular part into pure imaginary
        x_tilde = tf.matmul(F, x_tilde)        
        
        # calculate the filter function
        x_tilde = tf.square( tf.abs(x_tilde) )
        
        # retrieve the PSD of noise
        S = self.S
        
        # add extra dimension for batch
        S = tf.expand_dims(S, 0)
        
        # repeat the matrix along the batch dimension
        S = tf.tile( S, temp_shape )
        
        # multiply by the PSD of noise
        x_tilde = tf.multiply(S, x_tilde[:,0:self.N//2,:])
        
        # retrieve the trapezoidal matrix
        D = tf.constant(self.D, dtype=tf.float32)
        
        # add extra dimension for batch
        D = tf.expand_dims(D, 0)
        
        # repeat the matrix along the batch dimension
        D = tf.tile( D, temp_shape )
        
        # approximate the integral using the trapezoidal rule
        W = tf.matmul(D, x_tilde)      
        W = tf.reduce_sum(W , 1)*(self.fs/self.N)
        
        return tf.exp(-W)
###############################################################################
class QNS_AS:
    """
    This is class to esitmate the PSD of noise via Alvarez-Suter
    """
    def __init__(self, N, fs):
        """
        class constructor
        
        N : Total number of samples
        fs: Sampling frequency 
        """
        
        # store local variables
        self.N  = N
        self.fs = fs
        
        # a new tensorflow input layer that takes the switching waveform y(tau) as input
        input_switching_wfm = layers.Input(shape=(N,1), name="input_switching_wfm")
        
        # add a custom tensor layer to calculate the coherence function
        w = coherence_Layer(fs)(input_switching_wfm)
        
        self.model = Model(inputs = input_switching_wfm, outputs = w)
        
        # compile the model
        self.model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
        
        self.model.summary()
        self.training_history = []
    
    def train_model(self, training_x, training_y, epochs):
        """
        This method is for training the model to estimate the PSD of noise
        """
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.training_history = self.training_history + self.model.fit(training_x, training_y, epochs=epochs, batch_size=training_x.shape[0],verbose=2).history['loss']
     
    def predict_PSD(self):
        """
        This method is for predicting the PSD of noise after training
        """
        return self.model.get_weights()[0]
    
    def predict(self, training_x):
        """
        This function is to predict the coherence using the trained model
        
        """
        return self.model.predict(training_x)