"""
This module is for training the model using a dataset
"""
from qubitmlmodel import qubitMLmodel
import pickle

if __name__ == '__main__':  
    # 1) Load dataset
    f = open("./../datasets/CPMG_G_X_28.ds", 'rb')
    data = pickle.load(f)
    f.close()  
    
    # 2) Load all variables  
    T                = data["T"]
    M                = data["M"]
    Omega            = data["Omega"]
    training_inputs  = data["training_inputs"]
    training_targets = data["training_targets"]
    testing_inputs   = data["testing_inputs"]
    testing_targets  = data["testing_targets"]
    
    # 3)  Define the ML model
    mlmodel = qubitMLmodel(T/M, Omega, "Single_Axis", 2)
    
    # 4) Perform training 
    mlmodel.train_model_val(training_inputs, training_targets, testing_inputs, testing_targets, 3000)    
    
    # 5) Save results        
    mlmodel.save_model("trained_model_CPMG_G_X_28_3000")
#########################################################
    
    # 1) Load dataset
    f = open("./../datasets/CPMG_S_X_28", 'rb')
    data = pickle.load(f)
    f.close()  
    
    # 2) Load all variables  
    T                = data["T"]
    M                = data["M"]
    Omega            = data["Omega"]
    training_inputs  = data["training_inputs"]
    training_targets = data["training_targets"]
    testing_inputs   = data["testing_inputs"]
    testing_targets  = data["testing_targets"]
    
    # 3)  Define the ML model
    mlmodel = qubitMLmodel(T/M, Omega, "Single_Axis", 2)
    
    # 4) Perform training 
    mlmodel.train_model_val(training_inputs, training_targets, testing_inputs, testing_targets, 3000)    
    
    # 5) Save results        
    mlmodel.save_model("trained_model_CPMG_S_X_28_3000")
    
