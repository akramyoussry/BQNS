"""
This module is for implementing different applications using the trained model
"""
# preample
from qubitmlmodel import qubitMLmodel
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
###############################################################################        
if __name__ == '__main__':
    
    # 1) load the dataset
    f = open("./../datasets/CPMG_G_X_28.ds", 'rb')
    data = pickle.load(f)
    f.close()  
    T                = data["T"]
    M                = data["M"]
    Omega            = data["Omega"]
    training_inputs  = data["training_inputs"]
    training_targets = data["training_targets"]
    testing_inputs   = data["testing_inputs"]
    testing_targets  = data["testing_targets"]    
    time_range       = np.array([(0.5*T/M) + (j*T/M) for j in range(M)]) # time_domain vector
    
    # 2) load the trained model
    mlmodel = qubitMLmodel(T/M, Omega, "Single_Axis", 2)    
    mlmodel.load_model("trained_model_CPMG_G_X_28_3000")
###############################################################################    
    # 3) do quantum control 
    n_max = 28
     
    G_names= ["I", "X", "Y", "Z", "H", "pi_4"]     
    G = [np.array([[1.,0.],[0.,1.]]), np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0],[0.,-1.]]), np.array([[1.,1.],[1.,-1.]])/np.sqrt(2),expm(-1j*np.pi*0.25*np.array([[0.,1.],[1.,0.]])/2) ]
    
    for idx_G, G in enumerate(G):
        mlmodel.construct_controller(T, M, n_max)
        pulses = mlmodel.train_controller(G,1000)
    
        # plot the results
        plt.figure(figsize=[8, 6])
        plt.subplot(2,1,1)
        plt.plot(time_range, pulses[1][0,:,0],'r',label='h_x(t)')
        plt.xlabel('t',fontsize=11)
        plt.ylabel("$\mathbf{h}(t)$",fontsize=11)
        plt.grid()
        plt.legend(fontsize=11)
        plt.savefig("./../imgs/control_pulses_%s.pdf"%G_names[idx_G],format='pdf', bbox_inches='tight') 
        print("Fidelities for gate %s are %f %f, %f, %f\n"%(G_names[idx_G], 100*(1-mlmodel.controller_training_history["Fid_0_loss"][-1]), 100*(1-mlmodel.controller_training_history["Fid_1_loss"][-1]), 100*(1-mlmodel.controller_training_history["Fid_2_loss"][-1]), 100*(1-mlmodel.controller_training_history["Fid_3_loss"][-1])))

###############################################################################
    # 4) perform the spectrum analyis using Alvarez-Suter
    S = mlmodel.estimate_spectrum(T,M, n_max)
    
    # plot the actual and predicted spectra
    f      = np.fft.fftfreq(M)*M/T
    alpha  = 1
    S_Z    = np.array([(1/(fq+1)**alpha)*(fq<50) + (1/40)*(fq>50) + 0.8*np.exp(-((fq-20)**2)/10) for fq in f[f>=0]]) # desired single side band PSD
    
    f = f[f>=0]
    plt.figure(figsize=[4.8, 3.8])
    plt.plot(f[0:n_max], S_Z[0:n_max], label = "Actual")
    plt.plot(f[0:n_max], 2*S[-1][0:n_max], "r.", label="Predicted")
    plt.xlabel('$f$', fontsize=11)
    plt.ylabel(r'$|S_Z(f)|$',fontsize=11)
    plt.grid(True, which="both")
    plt.xlim([-10,n_max])
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig('./../imgs/QNS.pdf',format='pdf', bbox_inches='tight')

	# calculate and plot the coherence
    f      = np.fft.fftfreq(M)*M/T
	coherence_estimate = np.zeros((n_max,1))
    coherence_theory   = np.zeros((n_max,1))	 
    for n in range(n_max):
        t     = np.array([0] + [(k-0.5)/n for k in range(1,n+1)] + [1])*T # CPMG
        y     = np.array([sum([((-1.0)**k)*(tau>=t[k])*(tau<t[k+1]) for k in range(n+1)]) for tau in time_range])
        Y     = np.fft.fft(y)/(M/T)
        coherence_estimate[n,:] = np.exp(-np.trapz((abs(Y[f>=0])**2) *S[-1].T, f[f>=0])) 
        coherence_theory[n,:]   = np.exp(-np.trapz((abs(Y[f>=0])**2) * 0.5*S_Z, f[f>=0])) 
	
    plt.figure()
    plt.plot(S[3], '.',label="Estimated coherence from model")
    plt.plot(coherence_estimate, label="Theroetical coherence from estimated PSD ")
    plt.plot(coherence_theory, label="Theoretical coherence from theoretical PSD")
    plt.grid()
    plt.xlabel("n", fontsize=11)
    plt.ylabel("coherence", fontsize=11)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("./../imgs/coherence.pdf", bbox_inches='tight')

