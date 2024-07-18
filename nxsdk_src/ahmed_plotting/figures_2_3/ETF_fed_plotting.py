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




# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
        
        
if __name__ == "__main__":        
    # Time duration for the execution
    runTime = 60000
    
    # Number of excitatory neurons
    numECores = 128
    # Number of inhibitory neurons
    numICores = 64

    # Refractory period of the input spikes
    tEpoch = 1

    # An array of spiking probabilities of the input spike generators
    spikingProbability = [(0.4*n**2) for n in np.arange(0, 1, 0.03)]

    # Spiking probabilitoes of the noise generators
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1

    # Number of different frequencies of the input spikes
    numProb = len(spikingProbability)
    
    # Voltage decay time constant
    tau = 16
    
    # Recurrent efficacies of the excitatory connections for which the ETF is measured for
    Jee = [5, 15, 21, 22]
    len_Jee = len(Jee)

    # Normalizing
    # Normalized_Jee = [round(((i*(2**6))/11520), 3) for i in Jee] 
    Normalized_Jee = [round( ((i)*64/11520) , 3) for i in Jee]

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
        legend_labels.append(r"$J_{E,E} =$" + str(Normalized_Jee[ii]))


    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)

    df = pd.read_excel(script_directory + '/ETF_weights_final.xlsx')

    # Display the DataFrame
    meanFreqInArray = df[['FreqIn0', 'FreqIn1', 'FreqIn2', 'FreqIn3']].values
    meanFreqOutArray = df[['FreqOutExc0', 'FreqOutExc1', 'FreqOutExc2', 'FreqOutExc3']].values
    variance_out = df[['Var0', 'Var1', 'Var2', 'Var3']].values
    meanFreqOutArrayInh = df[['FreqOutInh0', 'FreqOutInh1', 'FreqOutInh2', 'FreqOutInh3']].values

    # Plot
    fig1 = plt.figure(1001, figsize=(25,25))
    plt.xlabel('Pre-spike rate [/100 timesteps]', fontsize = 35+size)
    plt.ylabel('Post-spike rate [/100 timesteps]', fontsize = 35+size)
    # plt.title('Effective Transfer Function', fontsize = 30)

    plt.tick_params(axis='both', which='major', labelsize=20+size)
    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'k']
    
    plt.plot(list(range(0, 40)), list(range(0,40)), color = plot_color[ii+1])

    for ii in range(len(Jee)):
        if ii > 1:
            f = interp1d(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], kind='linear')
            
            # Define x values for y=x line
            x_line = np.linspace(np.min(meanFreqInArray[:, ii]), np.max(meanFreqInArray[:, ii]), 10000)
            
            # Find intersection points
            idx = np.argwhere(np.diff(np.sign(f(x_line) - x_line))).flatten()

            if ii == 2:
                idx = idx[0:2]

            # print(idx)
            plt.plot(x_line[idx], f(x_line[idx]), 'o', markersize=30+size, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k', label='_nolegend_')
            plt.plot(x_line[idx[1]], f(x_line[idx[1]]), 'x', markersize=20+size, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k', label='_nolegend_')


        # Effective TF 
        if ii == 2:
            plt.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii], linewidth = 5)

        else:
            plt.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii])

        
        # Standard deviation
        plt.errorbar(meanFreqInArray[:, ii], 
                     meanFreqOutArray[:, ii], 
                     yerr=np.sqrt(variance_out[:, ii]), 
                     fmt='o', 
                     capsize=10, 
                     color='black', 
                     ecolor= plot_color[ii])
    
    
    
    legend_labels = ["Stability line"] + legend_labels
    plt.legend(legend_labels, fontsize = 20+size, loc='upper right')
    # Stability line
    plt.plot(13, 13, 'o', markersize=50+size, markerfacecolor='none', markeredgewidth=2, markeredgecolor='r', label='Intersection')

    plt.xlim(-1, 35)
    plt.ylim(-1, 35)

    plt.grid(True)

    # ZOOMED INSET
    bbox = transforms.Bbox.from_bounds(-157, 550, 1500, 1500)
    axins = inset_axes(plt.gca(), width="30%", height="30%", bbox_to_anchor=bbox, loc='center')

    for ii in range(len(Jee)):

        if ii > 1:
            f = interp1d(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], kind='linear')
            
            # Define x values for y=x line
            x_line = np.linspace(np.min(meanFreqInArray[:, ii]), np.max(meanFreqInArray[:, ii]), 10000)
            
            # Find intersection points
            idx = np.argwhere(np.diff(np.sign(f(x_line) - x_line))).flatten()

            if ii == 2:
                idx = idx[0:2]

            # print(idx)
            plt.plot(x_line[idx], f(x_line[idx]), 'o', markersize=30+size, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k', label='_nolegend_')
            plt.plot(x_line[idx[1]], f(x_line[idx[1]]), 'x', markersize=20+size, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k', label='_nolegend_')


        # Effective TF 
        if ii == 2:
            plt.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii], linewidth = 5)

        else:
            plt.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii])
        # Standard deviation
        plt.errorbar(meanFreqInArray[:, ii], 
                     meanFreqOutArray[:, ii], 
                     yerr=np.sqrt(variance_out[:, ii]), 
                     fmt='o', 
                     capsize=10, 
                     color='black', 
                     ecolor= plot_color[ii])
   
    # Stability line
    plt.plot(list(range(0, 40)), list(range(0,40)), color = plot_color[ii+1])

    # Set limits for inset axes
    axins.set_xlim(-1, 4)  
    axins.set_ylim(-1, 4)
    axins.set_xticks(np.arange(-1, 5, 2))
    axins.set_yticks(np.arange(-1, 5, 2))
    axins.set_xticks([0, 2, 4])
    axins.set_yticks([0, 2, 4])
    axins.tick_params(axis='both', which='major', labelsize=20+size, width = 3)
    # plt.xlabel('Pre-spike rate [/100]', fontsize = 30)
    # plt.ylabel('Post-spike rate [/100]', fontsize = 30)
    plt.title('Zoomed inset', fontsize = 30+size)
    # Add grid to inset axes (optional)
    axins.grid(True)
    # plt.tight_layout()

    filename1 = "final"    
    # Save the figure
    
    #if haveDisplay:
    #    plt.show()
    #else:
    fileName = script_directory + "/figures/ETF_weights_"+filename1+".pdf"
    print("No display available, saving to file " + fileName + ".")
    fig1.savefig(fileName)
