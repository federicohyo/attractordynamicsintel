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
    
    script_path = os.path.abspath(__file__)

    # Get the directory of the current script
    script_directory = os.path.dirname(script_path) + '/'

    filename1 = "multi_ahm__final"
    weightMatrix = []

    for time in range(8):
        weightMatrix.append(np.load(script_directory + "stdp_multi_retrial/output_data/weightmatrix" + str(time) +filename1+".npy"))
    
    weight_ave = [[] for _ in range(len(weightMatrix))] # 8 time lists
    weight_std = [[] for _ in range(len(weightMatrix))]
    # weight_ave = []
    # weight_std = []
    for time in range(8):
        for pop in range(4):
            weight_ave[time].append(np.nanmean(weightMatrix[time][pop*128:(pop+1)*128, pop*128:(pop+1)*128]))
            weight_std[time].append(np.nanstd(weightMatrix[time][pop*128:(pop+1)*128, pop*128:(pop+1)*128]))
        
    print(weight_ave)
    scale = (64/11520)
    # scale = 1
    for i, arr in enumerate(weight_ave):
        weight_ave[i] = [num.astype(float) * scale for num in arr]
    
    for i, arr in enumerate(weight_std):
        weight_std[i] = [num.astype(float) * scale for num in arr]
    dt = 30600//8
    print("Time  | Avg Wgt. of Subpopulations")
    for time in range(1,8,1):
        string = str((time+1)*dt) + " & "
        for pop in range(4):
            string = string + str(round(weight_ave[time][pop], 3)*10) + " $\pm$"
            string = string + str(round(weight_std[time][pop], 3)*10) + " & "
        print(str(string)+"\\")  

    return


if __name__ == "__main__":
    main()