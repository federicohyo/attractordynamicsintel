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
import random
import math

haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
    
        
if __name__ == "__main__": 
    total_accuracy = []
    accuracy = []
    errorPercentage = []
    percentages = ["0", "20.0", "40.0", "55.0", "65.0", "70.0", "75.0", "80.0", "90.0"]
    for percentage in percentages:
        filename = 'data/error_correction_percentage' + percentage + '.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                (total_accuracy0, accuracy0, errorPercentage0) = pickle.load(file)
        if percentage == "0":
            accuracy = np.matrix(accuracy0)
        else:
            accuracy = np.concatenate((accuracy, np.matrix(accuracy0)), axis = 0)
        '''if os.path.exists(filename):
            with open(filename, 'rb') as file:
                (total_accuracy1, accuracy1, errorPercentage1) = pickle.load(file)
            print(total_accuracy1)
            if total_accracy0 == -1:
                total_accuracy0 = total_accuracy1
                accuracy0 = accuracy1
                errorPercentage0 = errorPercentage1
            else:
                total_accuracy0 = (total_accuracy0+2*total_accuracy1)/3
                accuracy0 = (accuracy0 + 2*accuracy1)/3'''
         
        total_accuracy.append(total_accuracy0)
        errorPercentage.append(errorPercentage0) 
    print(accuracy)
    hamming_distance = 128*0.01*np.array(errorPercentage)
    fig = plt.figure(1001)
    plt.plot(hamming_distance, total_accuracy, '-o')
    for ii in range(4):
        plt.plot(hamming_distance, accuracy[:, ii], '-o')
    plt.legend(["average", "attractor1", "attractor2", "attractor3", "attractor4"], fontsize = 14)    
    plt.xlabel("hamming distance", fontsize = 15)
    plt.ylabel("reconstruction accuracy [%]", fontsize = 15)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/error_correction/accuracy.png"
        fig.savefig(fileName)
        print("No display available, saved to file " + fileName + ".")