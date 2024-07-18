# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import sys
import matplotlib as mpl
from nxsdk.utils.plotutils import plotRaster
import time
from nxsdk.graph.monitor.probes import *
import pandas as pd
import matplotlib.colors as colors
import pickle


haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')    
    
if __name__ == "__main__":        
    with open('data/data_weight_multi_Overlap.pkl', 'rb') as file:
        weightMatrix = pickle.load(file)
    
    runTime = 80300
    numESubPop = 4
    numESubCores = 128
    dt = runTime//4
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

    fig2 = plt.figure(1003, figsize=(40, 40))    
    import matplotlib.colors as colors
    cmap = colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'black'])
    for time in range(4):
        ax = plt.subplot(2, 2, time+1)
        # Create a heatmap using matplotlib
        plt.imshow(weightMatrix[time], cmap=cmap, interpolation='nearest', vmin=0, vmax=86)
        cbar = plt.colorbar(shrink=0.9)
        cbar.ax.tick_params(labelsize=55)
        plt.title("t="+str((time+1)*dt), fontsize=60)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)

    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/overlap/SynapticMatrices.png"
        print("No display available, saving to file " + fileName + ".")
        fig2.savefig(fileName)
    #-------------------------------------------------------------------------     
    # Plot probability distribution
    #-------------------------------------------------------------------------   
    fig = plt.figure(1002, figsize=(40, 40))   
    for time in range(4):
        ax = plt.subplot(2, 2, time+1)
        for pop in range(numESubPop):
            array = np.ravel(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                pop*numESubCores:(pop+1)*numESubCores])
            mask = ~np.isnan(array)
            array = array[mask]
            # Plot the histogram
            plt.hist(array, bins=21, density=True, histtype='step', stacked=True, fill=False)
        if time == 2 or time == 3:
            plt.xlabel('Efficacy', fontsize=60)
        if time == 0 or time == 2:
            plt.ylabel('Probability density', fontsize=60)
        plt.title("t="+str((time+1)*dt), fontsize=60)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)
        plt.legend(["attractor 1", "attractor 2", "attractor 3", "attractor 4"], fontsize=45)
        plt.ylim(0, 0.1)
    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/overlap/SynapticMatrices_Distribution.png"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)