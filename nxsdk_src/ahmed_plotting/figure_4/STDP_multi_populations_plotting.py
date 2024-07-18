# Import modules

import matplotlib.pyplot as plt
# import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
# from nxsdk.utils.plotutils import plotRaster
# from nxsdk.graph.monitor.probes import *
import pandas as pd
import matplotlib.colors as colors
import pickle
import string

# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

def main():

    # Time duration of the execution
    runTime = 30600#30300#30260 
    print("The total number of timesteps is ", runTime, ".")
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = [0.05, 0.6] # [low frequency input, high frequency input]
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1 #0.01 #xx
    #inputFraction = 3
    # Number of neurons in each excitatory subpopulation
    numESubCores = 128
    # Number of excitatory subpopulations
    numESubPop = 4#4
    # Number of excitatory neurons
    numECores = numESubCores*numESubPop
    # Number of inhibitory neurons
    numICores = numECores//2
    # Number of pixels
    numSpikeGen = numECores
    # Time constant
    tau=16
    # Refractory period of the input spikes
    tEpoch = 0#1 #xx
    repetition = (runTime-300)//3000#6000 #xx no difference
    # Initial recurrent weight (learning)
    Jee = 0
    learning = 1
    
    # ImpParam = [20, 20]  #max 55 ##################### 4 populations #################
    # x1Tau=20
    # y1Tau=10
    # tEpochlr = 19

    ImpParam = [15, 15]#[25]
    x1Tau=18
    y1Tau=10
    tEpochlr = 19

    # Sample cycle of weights
    dt = runTime//8 #xx
    
    GENERATE = False
    plotting = True


    if (runTime == 30600):
        # load = "_30_4"
        load = "__final"

    elif (runTime == 30300):
        load = "_30" # === 2 pop good
    else:
        load = "lava"

    print(f"E - cores: {numECores}")
    print(f"Jee = {Jee}")
    print(f"learning: {learning}")
    print(f"ImpParam = {ImpParam}")
    print(f"x1Tau = {x1Tau}")
    print(f"y1Tau = {y1Tau}")
    print(f"tEpochlr = {tEpochlr}")
    # Generate spike times

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path) + '/'


    print("loading spikes from disk")
   
    spikeTimes = []
    with open(script_directory + "data/stdp_multi_retrial/spiketimes/spikeTimes" + "_30_4" + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeTimes.append(sublist)

    spikeResetTimes = []
    with open(script_directory + "data/stdp_multi_retrial/spiketimes/spikeResetTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeResetTimes.append(sublist)

    spikeENoiseTimes = []
    with open(script_directory + "data/stdp_multi_retrial/spiketimes/spikeENoiseTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeENoiseTimes.append(sublist)

    spikeINoiseTimes = []
    with open(script_directory + "data/stdp_multi_retrial/spiketimes/spikeINoiseTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeINoiseTimes.append(sublist)
    
    for trial in range(1,2):
        # Initialize a network
        # Learning parameters
        x1Imp=ImpParam[0]
        y1Imp=ImpParam[1]
        
        print("Spikes were generated/or loaded. Configuring network...")    



        print("Loading output data")
        # filename1 = "multi_ahm_new_3"
        filename1 = "multi_ahm__final"


        cg1Spikes = []
        with open(script_directory + "data/stdp_multi_retrial/output_data/cg1Spikes_STDP"+filename1+".txt", "r") as f:
            for line in f:
                sublist = list(map(int, line.strip().split()))
                cg1Spikes.append(sublist)
        
        cg1_Voltage = np.load(script_directory + "data/stdp_multi_retrial/output_data/cg1_Voltage"+filename1+".npy")
        meanFreqOutArray = np.load(script_directory + "data/stdp_multi_retrial/output_data/meanFreqOutArray_STDP"+filename1+".npy")
        # weightMatrix = np.load("data/stdp_multi_retrial/output_data/weightmatrix"+filename1+".npy")
        # Shape weight matrices 


        weightArray = [np.array([]) for _ in range(runTime//dt)]
        print(filename1)
        # filename1 = "multi_ahm_new"
        filename1 = "multi_ahm__final"
        weightMatrix = []
        weight_ave = [[] for _ in range(len(weightArray))] # 8 time lists
        weight_std = [[] for _ in range(len(weightArray))]
        
        # with open('data/stdp_multi_retrial/output_data/weightmatrix_3_FINAL.pkl', 'rb') as file:
        #         weightMatrix = pickle.load(file)
        for time in range(8):
            weightMatrix.append(np.load(script_directory + "data/stdp_multi_retrial/output_data/weightmatrix" + str(time) +filename1+".npy"))

            for pop in range(numESubPop):
                weight_ave[time].append(np.nanmean(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))
                weight_std[time].append(np.nanstd(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))
        # print(weight_ave)          
        print("Time  | Avg | Std | Wgt. of Subpopulations")

        for time in range(len(weightArray)):
            string1 = str((time+1)*dt) + " | "
            for pop in range(numESubPop):
                string1 = string1 + str(round(weight_ave[time][pop])) + " | "
                string1 = string1 + str(round(weight_std[time][pop])) + " | "
            print(str(string1))

        #weight matrix


        if (plotting):

            # Process and plot pre,  postsynaptic spikes, membrane voltage and excitatory frequency over time
            fig = plt.figure(1001, figsize=(30, 20))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']

            ticks = 35
            label = 40
            letter = 50
            title = 45
            ax0 = plt.subplot(3, 1, 1)
            plt.eventplot(spikeTimes, color='black',  linewidths=0.7, rasterized = True) #512
            plt.title('Presynaptic spikes', fontsize = title)
            # plt.xticks(fontsize=60)
            # plt.yticks(fontsize=60)
            # plt.ylabel("neuron index", fontsize = 80)
            #for i in range(len(spikeTimes)):
            #    plt.plot(spikeTimes[i], np.repeat(i,len(spikeTimes[i])), marker=".", markersize=0.1, color='black') #512
            #plt.title('presynaptic spikes', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30600,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=ticks)
            plt.yticks(fontsize=ticks)
            ax0.set_ylim([0,512])
            plt.ylabel("Neuron ID", fontsize = label)
            ax0.text(0, 1.03, string.ascii_uppercase[0], transform=ax0.transAxes, size=letter, weight='bold')




            cg1Spikes_raster = []
            for sublist in cg1Spikes:
                indexes = [i for i, value in enumerate(sublist) if value == 1]
                cg1Spikes_raster.append(indexes)



            ax1 = plt.subplot(3, 1, 2)
            # cg1Spikes[0].plot() #512
            plt.eventplot(cg1Spikes_raster, color='black', linewidths=0.5, rasterized = True)
            plt.title('Postsynaptic spikes', fontsize = title)
            # plt.xticks(fontsize=60)
            # plt.yticks(fontsize=60)
            # plt.xlabel("", fontsize = 60)
            # plt.ylabel("neuron index", fontsize = 80)
            #for i in range(len(cg1Spikes_raster)):
            #    plt.plot(cg1Spikes_raster[i], np.repeat(i,len(cg1Spikes_raster[i])), marker=".", markersize=0.1, color='black') #512
            #plt.title('postsynaptic spikes', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30600,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=ticks)
            plt.yticks(fontsize=ticks)
            plt.xlabel("", fontsize = label)
            plt.ylabel("Neuron ID", fontsize = label)
            ax1.set_ylim([0,512])
            ax1.text(0, 1.03, string.ascii_uppercase[1], transform=ax1.transAxes, size=letter, weight='bold')


            # ax2 = plt.subplot(4, 1, 3)
            # for pop in range(numECores//128):
            #     plt.plot(cg1_Voltage[pop])
            #     # v1 = cg1_Voltage[pop].plot() #4
            # plt.title('membrane voltage', fontsize = 100)
            # plt.xticks(fontsize=60)
            # plt.yticks(fontsize=60)
            # plt.xlabel("", fontsize = 60)
            # plt.ylabel("voltage", fontsize = 80)

            ax0.set_xlim(0, runTime)
            ax1.set_xlim(ax0.get_xlim())
            # ax2.set_xlim(ax0.get_xlim())
            

            ax4 = plt.subplot(3, 1, 3)
            for pop in range(numESubPop):
                ax4.plot(meanFreqOutArray[pop], color=colors[pop], label="subpop "+str(pop+1), linewidth = 3)
            plt.xlabel("Timesteps", fontsize = label)
            plt.ylabel("Freq. [/100 timesteps]", fontsize =label)
            plt.title('Spiking activity of excitatory population over time', fontsize = title)
            # plt.xticks(fontsize=60)
            # plt.yticks(fontsize=60)
            plt.xticks(ticks=np.linspace(0,30600,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=ticks)
            plt.yticks(fontsize=ticks)
            plt.legend(loc="best", fontsize=25)
            ax4.text(0, 1.03, string.ascii_uppercase[2], transform=ax4.transAxes, size=letter, weight='bold')


            ax4.set_xlim(ax0.get_xlim())
            # ax4.axhline(y=12.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=13, color='black', linestyle='--', linewidth=5)
            ax4.axhline(y=14, color='black', linestyle='--', linewidth=4)
            # ax4.axhline(y=14, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=14.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=15, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=15.5, color='black', linestyle='--', linewidth=5)


            plt.tight_layout()
            if haveDisplay:
                plt.show()
            else:
                fileName = script_directory + "figures/stdp/PSTH_plot" +  filename1 + ".pdf" #.pdf
                print("No display available, saving to file " + fileName + ".")
                fig.savefig(fileName, dpi=500)



if __name__ == "__main__":        
    main()
