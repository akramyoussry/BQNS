"""
This module does all the plots for the performance analysis of the trained models 
"""
# preample
from qubitmlmodel import qubitMLmodel
import pickle
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool 
###############################################################################
def process_dataset(dataset_name):
    """
    This function processes a dataset and extracts all required plots
    
    dataset_name: Name of the dataset following our convention
    """

    print("Proceesing Dataset %s\n"%dataset_name)
    
    # 1) load the dataset
    f = open("./../datasets/%s.ds"%dataset_name, 'rb')
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
    if testing_inputs[1].shape[-1] == 1:
        mlmodel = qubitMLmodel(T/M, Omega, "Single_Axis", testing_inputs[0].shape[-1])
    else:
        mlmodel = qubitMLmodel(T/M, Omega, "Multi_Axis", testing_inputs[0].shape[-1]//2)    
    mlmodel.load_model("trained_model_%s_3000"%dataset_name)
    
    # 3) plot the MSE    
    plt.figure(figsize=[4.8, 3.8])
    plt.loglog(mlmodel.training_history, label="Training")
    plt.loglog(mlmodel.val_history, label="Validation")
    plt.legend(fontsize=11)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('MSE',fontsize=11)
    plt.xscale('log')
    plt.xticks(sum([[i*j for i in range(1,11)] for j in [1,10,100,1000]],[]),fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, which="both")
    plt.savefig("./../imgs/model_training_%s.pdf"%dataset_name, bbox_inches='tight')
    
    # 4) Evaluate MSE over the training and testing sets
    total_MSE = (mlmodel.model.evaluate(training_inputs, training_targets, batch_size = training_targets.shape[0])[0],
                 mlmodel.model.evaluate(testing_inputs , testing_targets , batch_size = testing_targets.shape[0])[0])
    
    # 5) Evaluate the MSE per each testing example for statistical plots    
    MSE = [mlmodel.model.evaluate([testing_inputs[0][idx:idx+1,:], testing_inputs[1][idx:idx+1,:]], testing_targets[idx:idx+1,:], batch_size = 1, verbose=0)[0] for idx in range(testing_targets.shape[0])]

    # 6) Plot best, average, and worst case examples
    worst_example   = np.argmax(MSE)
    best_example    = np.argmin(MSE)
    average_example = np.argmin(np.abs(MSE-np.average(MSE)))
    
    predicted_measurements_testing  = mlmodel.predict_measurements(testing_inputs)
    
    for idx, example in enumerate( [worst_example, average_example, best_example] ):
        plt1 = testing_inputs[1][example,:]
        plt2 = testing_targets[example,:]
        plt3 = predicted_measurements_testing[example,:]
        
        plt.figure(figsize=[8, 6])
        plt.subplot(2,1,1)
        if testing_inputs[1].shape[-1] == 1:
            plt.plot(time_range, plt1[:,0],'r',label='h_x(t)')
        else:
            plt.plot(time_range, plt1[:,0],'r',label='h_x(t)')
            plt.plot(time_range, plt1[:,1],'k',label='h_y(t)')
        plt.xlabel('t',fontsize=11)
        plt.ylabel("$\mathbf{h}(t)$",fontsize=11)
        plt.grid()
        plt.legend(fontsize=11)
            
        ax = plt.subplot(2,1,2)
        plt.semilogy((plt2-plt3)**2, "D")
        plt.xlabel('Initial State/Measurement Operator',fontsize=11)
        plt.xticks(fontsize=11)
        labels = [ [r"$\rho_{%s}, \sigma_{%s}$"%(rho,O) for O in ["x","y","z"]] for rho in ["x+","x-","y+","y-","z+","z-"]]
        labels = sum(labels,[])
        ax.set_xticks([x for x in range(18)])
        ax.set_xticklabels(labels, rotation=90)
        plt.ylim([-1.1, 1.1])
        plt.ylabel("Prediction Error Squared",fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(which="both") 
    
        plt.tight_layout()
        plt.savefig('./../imgs/ex_%s_%d.pdf'%(dataset_name,idx),format='pdf', bbox_inches='tight')
        
    return MSE, total_MSE
##############################################################################        
if __name__ == '__main__':
       
    # 1) specify the datasets we want to analyze their results
    datasets = ["CPMG_G_X_28", "CPMG_S_X_28", "CPMG_G_XY_7", "CPMG_G_XY_pi_7", "CPMG_G_XY_7_nl", "CPMG_G_XY_pi_7_nl"]
    
    # 2) do the analysis in parallel
    with Pool() as p:
        results = p.map(process_dataset, datasets)
        
    MSE       = [x[0] for x in results]
    total_MSE = [x[1] for x in results]
    
    # 3) display MSE for all datasets
    for idx,_ in enumerate(datasets):
        print("%e,%e\n"%total_MSE[idx])
    
    # 4) plot the comparitve plots
    plt.figure()
    plt.boxplot([np.log10(x) for x in MSE])
    ax = plt.gca()
    ax.set_yticks(np.arange(-7, 0))
    ax.set_yticklabels(10.0**np.arange(-7, 0))
    ax.set_xticklabels(datasets)
    plt.grid()
    plt.xlabel("Dataset",fontsize=11)
    plt.xticks(rotation = 45, fontsize=11)
    plt.ylabel("MSE", fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig('./../imgs/boxplot.pdf', format='pdf', bbox_inches='tight')
    
    plt.figure()
    plt.violinplot([np.log10(x) for x in MSE], showmeans=False,showmedians=True)
    ax = plt.gca()
    ax.set_yticks(np.arange(-7, 0))
    ax.set_yticklabels(10.0**np.arange(-7, 0))
    plt.setp(ax, xticks=[idx+1 for idx in range(len(MSE))], xticklabels=datasets)
    plt.grid()
    plt.xlabel("Dataset",fontsize=11)
    plt.xticks(rotation = 45, fontsize = 11)
    plt.ylabel("MSE", fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig('./../imgs/violinplot.pdf', format='pdf', bbox_inches='tight')
    
    # 5) plot the PSD
    T      = 1
    M      = 4096
    f      = np.fft.fftfreq(M)*M/T
    alpha  = 1
    S_Z    = np.array([(1/(fq+1)**alpha)*(fq<50) + (1/40)*(fq>50) + 0.8*np.exp(-((fq-20)**2)/10) for fq in f[f>=0]])  # desired single side band PSD
    alpha  = 1.5
    S_X    = np.array([(1/(fq+1)**alpha)*(fq<20) + (1/96)*(fq>20) + 0.5*np.exp(-((fq-15)**2)/10) for fq in f[f>=0]]) # desired single side band PSD
    
    plt.figure(figsize=[4.8, 3.8])
    plt.plot(f[f>=0], S_Z)
    plt.xlabel('$f$', fontsize=11)
    plt.ylabel(r'$|S_Z(f)|$',fontsize=11)
    plt.grid(True, which="both")
    plt.xlim([-50,500])
    plt.savefig('./../imgs/PSD_Z.pdf',format='pdf', bbox_inches='tight')
    
    plt.figure(figsize=[4.8, 3.8])
    plt.plot(f[f>=0], S_X)
    plt.xlabel('$f$', fontsize=11)
    plt.ylabel(r'$|S_X(f)|$',fontsize=11)
    plt.grid(True, which="both")
    plt.xlim([-50,500])
    plt.savefig('./../imgs/PSD_X.pdf',format='pdf', bbox_inches='tight')

    # 6) plot the monte carlo simulations
    
    state_descr = ["X+", "Y+", "Z+"]
    K=5000
    for idx_state in range(3):
        f = open("./../datasets/montecarlo_%d.ds"%idx_state, 'rb')
        data = pickle.load(f)
        f.close()
        expectations = data["expectations"]
        
        # plot the cumulative sum
        plt.figure(figsize=[4.8, 3.8])
        E = [np.cumsum(expectations[0:K,idx])/[k for k in range(1,K+1)] for idx in range(3) ]
        plt.plot(E[0], label=r'$X$')
        plt.plot(E[1], label=r'$Y$')
        plt.plot(E[2], label=r'$Z$')
        plt.xlabel('Number of Realizations', fontsize=11)
        plt.ylabel('Measurement outcomes for %s state'%state_descr[idx_state], fontsize=11)
        plt.grid()
        plt.legend(fontsize=11)
        plt.savefig('./../imgs/montecarlo_%d.pdf'%idx_state,format='pdf', bbox_inches='tight')
    
    f = open("./../datasets/montecarlo_pulses.ds", 'rb')
    data = pickle.load(f)
    f.close()
    h_x        = data["h_x"]
    h_y        = data["h_y"]
    time_range = data["time_range"]
    
    plt.figure(figsize=[8, 6])
    plt.subplot(2,1,1)
    plt.plot(time_range, h_x,'r',label='h_x(t)')
    plt.plot(time_range, h_y,'k',label='h_y(t)')
    plt.xlabel('t',fontsize=11)
    plt.ylabel("$\mathbf{h}(t)$",fontsize=11)
    plt.grid()
    plt.legend(fontsize=11)
    plt.savefig('./../imgs/montecarlo_pulses.pdf',format='pdf', bbox_inches='tight')  