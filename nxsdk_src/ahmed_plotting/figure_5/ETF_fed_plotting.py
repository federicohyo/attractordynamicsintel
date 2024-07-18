# Import modules
import matplotlib.pyplot as plt
# import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch
import matplotlib.transforms as transforms
import pickle




# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
        
        
def main():        
    # Time duration for the execution
    runTime = 60000
   
    # Number of excitatory neurons
    numECores = 128
    # Number of inhibitory neurons
    numICores = 64

    # Refractory period of the input spikes
    # tEpoch = 1

    # An array of spiking probabilities of the input spike generators
    spikingProbability = [(0.4*n**2) for n in np.arange(0, 1, 0.03)]

    # Spiking probabilitoes of the noise generators
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1

    # Number of different frequencies of the input spikes
    numProb = len(spikingProbability)
    
    # Voltage decay time constant
    tau = 16

    filename1 = "multi_ahm__final"
    weightMatrix = []

    for time in range(8):
        weightMatrix.append(np.load("stdp_multi_retrial/output_data/weightmatrix" + str(time) +filename1+".npy"))
    
    fig1, axs = plt.subplots(1, 3, figsize=(60, 20))
    for i in range(1,4): # 3 TIME STEPS
        dt = 30600//8
        
        if i == 1:
            time = 1
        elif i == 2:
            time = 3
        else: 
            time = 7
        
        len_Jee = 4

        size = 10
    
        # Variable initialization
        cg1Spikes = [[[] for _ in range(numProb)] for _ in range(len_Jee)] 
        meanFreqInArray = np.zeros((numProb, len_Jee))
        meanFreqOutArray = np.zeros((numProb, len_Jee))
        meanFreqOutArrayInh = np.zeros((numProb, len_Jee))
        variance_out = np.zeros((numProb, len_Jee))
        variance_outInh = np.zeros((numProb, len_Jee))
        legend_labels = []
        
        for ii in range(len_Jee):    
            # legend_labels.append(r"$J_{E,E} =$" + str(i+1))
            legend_labels.append("subpop "+ str(ii+1))

        # df = pd.read_excel('ETF_weights_final.xlsx')
        df = pd.read_excel('ETF_weights_i='+str((time+1)*dt)+'.xlsx')
        # Display the DataFrame
        meanFreqInArray = df[['FreqIn2', 'FreqIn1', 'FreqIn0', 'FreqIn3']].values
        meanFreqOutArray = df[['FreqOutExc2', 'FreqOutExc1', 'FreqOutExc0', 'FreqOutExc3']].values
        variance_out = df[['Var2', 'Var1', 'Var0', 'Var3']].values
        meanFreqOutArrayInh = df[['FreqOutInh2', 'FreqOutInh1', 'FreqOutInh0', 'FreqOutInh3']].values

        # Plot
        ax = axs[i-1]
        # fig1 = plt.figure(1001, figsize=(90,30))
        ax.set_xlabel('Pre-spike rate [/100 timesteps]', fontsize = 65)
        if time == 1:
            ax.set_ylabel('Post-spike rate [/100 timesteps]', fontsize = 65)
           
        #ax.set_title("Timestep "+str((time+1)*dt), fontsize=70)    
        if(time ==1):
            ax.set_title("Timestep 7,650", fontsize=70)
        if(time ==3):
            ax.set_title("Timestep 15,300", fontsize=70)
        if(time ==7):
            ax.set_title("Timestep 30,600", fontsize=70)

        ax.tick_params(axis='both', which='major', labelsize=50)
        plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'k']
        
        ax.plot(list(range(0, 40)), list(range(0,40)), color = plot_color[ii+1])

        for ii in range(len_Jee):
            # if ii == 2:
            f = interp1d(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], kind='linear')
            
            # Define x values for y=x line
            x_line = np.linspace(np.min(meanFreqInArray[:, ii]), np.max(meanFreqInArray[:, ii]), 10000)
            
            # Find intersection points
            idx = np.argwhere(np.diff(np.sign(f(x_line) - x_line))).flatten()


            ax.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii])

            
            # Standard deviation
            ax.errorbar(meanFreqInArray[:, ii], 
                        meanFreqOutArray[:, ii], 
                        yerr=np.sqrt(variance_out[:, ii]), 
                        fmt='o', 
                        capsize=10, 
                        color='black', 
                        ecolor= plot_color[ii])
        legend_labels = ["Stability line"] + legend_labels
        ax.legend(legend_labels, fontsize = 40, loc='upper left')
        # Stability line  
        ax.set_xlim(-1, 30)
        ax.set_ylim(-1, 25)

        ax.grid(True)

    plt.tight_layout(w_pad=8.0)
    filename1 = "__final"    
    # Save the figure
    
    #if haveDisplay:
    #    plt.show()
    #else:
    fileName = "ETF_weights_"+filename1+".pdf"
    print("No display available, saving to file " + fileName + ".")
    fig1.savefig(fileName)



if __name__ == "__main__":
    main()
