# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
# import nxsdk.api.n2a as nx
import os
import numpy as np
import sys
import matplotlib as mpl
# from nxsdk.utils.plotutils import plotRaster
import time
# from nxsdk.graph.monitor.probes import *
import pandas as pd
import matplotlib.colors as colors
import pickle


haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')    
    
def main():       
#     with open('data/stdp_multi_retrial/output_data/weightmatrix_3_FINAL.pkl', 'rb') as file:
#         weightMatrix = pickle.load(file)
    
    script_path = os.path.abspath(__file__)

    # Get the directory of the current script
    script_directory = os.path.dirname(script_path) + '/'

    weightMatrix = []
    filename1 = "multi_ahm__final"
    for time in range(8):
        weightMatrix.append(np.load(script_directory + "data/stdp_multi_retrial/output_data/weightmatrix" + str(time) +filename1+".npy"))

    
    NORMALIZE = True

    if NORMALIZE:
        scale_factor =(64/11520)#64/11520
    else: 
        scale_factor = 1

    weightMatrix_heat = [arr.astype(float) * scale_factor for arr in weightMatrix]
    # print(weightMatrix)
    # print("Scaled data sample:", weightMatrix[0][:5, :5])
    runTime = 30600
    numESubPop = 4
    numESubCores = 128
    dt = runTime//8
    weight_ave = [[] for _ in range(len(weightMatrix))]
    for time in range(len(weightMatrix)):
        for pop in range(numESubPop):
            weight_ave[time].append(np.nanmean(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                            pop*numESubCores:(pop+1)*numESubCores]))
    print("Time  | Avg Wgt. of Subpopulations")
    for time in range(len(weightMatrix)):
        string = str((time+1)*dt) + " | "
        for pop in range(numESubPop):
            string = string + str(round(weight_ave[time][pop])) + " | "
        print(str(string))

    fig2 = plt.figure(1003, figsize=(60, 60))    
    import matplotlib.colors as colors
    cmap = colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'black'])
    timesteps = [1, 3, 5, 7]
    for index, time in enumerate(timesteps):
        # time_index = 2*time+1
        ax = plt.subplot(2, 2, index+1)
        # Create a heatmap using matplotlib
        if NORMALIZE:
            vmin1 = 0
            vmax1 = 0.15
        else:
            vmin1 = 0
            vmax1 = 25
        plt.imshow(weightMatrix_heat[time], cmap=cmap, interpolation='nearest', vmin=vmin1, vmax=vmax1, rasterized=True)
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.tick_params(labelsize=45)
        plt.title("Timestep "+str((time+1)*dt), fontsize=70)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)
        if index == 0:
            plt.ylabel('Presynaptic neuron index ', fontsize=60)
        plt.xlabel('Postsynaptic neuron index ', fontsize=60)

    plt.tight_layout(w_pad=8.0)
    if haveDisplay:
        plt.show()
    else:
        fileName = script_directory + "figures/overlap/SynapticMatrices.pdf"
        print("No display available, saving to file " + fileName + ".")
        fig2.savefig(fileName,  dpi=150)
    #-------------------------------------------------------------------------     
    # Plot probability distribution
    #-------------------------------------------------------------------------   
    # if NORMALIZE:
    fig = plt.figure(1002, figsize=(60, 60))   
    for index, time in enumerate(timesteps):
        # time_index = 2*time+1
        ax = plt.subplot(2, 2, index+1)
        for pop in range(numESubPop):
            array = np.ravel(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                pop*numESubCores:(pop+1)*numESubCores])
            mask = ~np.isnan(array)
            array = array[mask]
            # Calculate histogram without density normalization
            hist, bins = np.histogram(array, bins=21, density=False)
        
            # Calculate the bin width
            bin_width = bins[1] - bins[0]

            # Normalize the histogram manually
            hist_density = hist / (len(array) * bin_width)
            # hist_density = hist * bin_width
            
            bin_centers = 0.5 * (bins[:-1] + bins[1:]) * scale_factor
            ax.plot(bin_centers, hist_density, drawstyle='steps-mid', linewidth=5, label=f'subpop {pop + 1}')


            # plt.hist(array, bins=21, density=True, histtype='step', stacked=True, fill=False, linewidth=4, label=f'attractor {pop + 1}')
        if index == 0:
            plt.ylabel('Probability density', fontsize=65)
        plt.xlabel('Efficacy', fontsize=65)

        # if time == 2 or time == 3:
        #     plt.xlabel('Efficacy', fontsize=60)
        # if time == 0 or time == 2:
        #     plt.ylabel('Probability density', fontsize=60)
        plt.title("Timestep "+str((time+1)*dt), fontsize=70)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)

        plt.legend(["subpop 1", "subpop 2", "subpop 3", "subpop 4"], fontsize=55)
        # plt.legend(loc='best')
        # plt.legend(fontsize=45, loc='best')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=55, loc='upper right' if time < 7 else 'upper left')

        x_ticks = ax.get_xticks()
        x_ticks = x_ticks[x_ticks != 0]
        ax.set_xticks(x_ticks)
        plt.ylim(0, 0.35)
        if NORMALIZE:
            plt.xlim(0, 0.15)
        else:
            plt.xlim(0, 27)
        plt.tight_layout(w_pad=8.0)

    if haveDisplay:
        plt.show()
    else:
        fileName = script_directory + "figures/overlap/SynapticMatrices_Distribution.pdf"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)

if __name__ == "__main__": 
    main()