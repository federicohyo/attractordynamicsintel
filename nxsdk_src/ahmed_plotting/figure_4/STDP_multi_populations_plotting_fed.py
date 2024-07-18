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

# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Create numSpikeGen spike generators
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) 
    return spikeGen

# Produce spike times of numSpikeGen spike generators
def genInputSpikeTimes(runTime, spikingProbability, tEpoch, numSpikeGen, numESubPop, repetition):
    runTimeRep = runTime//repetition
    spikeTimes = []
    initialIdleTime = 300
    spikes = np.zeros((numSpikeGen, runTime)) 
    numNeuronsPerSubpop = round(numSpikeGen/numESubPop)
    for pop in range(numESubPop):
        for gen in range(int(numNeuronsPerSubpop*pop), int(numNeuronsPerSubpop*(pop+1))):
            refCtr = 0
            flag_pre = 0
            flag = 0
            rep = 0
            for time in range(runTime):
                if time >= int(runTimeRep*(rep+pop/numESubPop))+initialIdleTime \
                              and time < int(runTimeRep*(rep+(pop+1/3)/numESubPop))+initialIdleTime:
                    spikes[gen][time] = (np.random.rand(1) < spikingProbability[1]) and refCtr <= 0 
                    flag = 1
                else:
                    spikes[gen][time] = (np.random.rand(1) < spikingProbability[0]) and refCtr <= 0 
                    flag = 0
                if flag == 0 and flag_pre == 1:
                    rep = rep + 1
                flag_pre = flag
                if spikes[gen][time]:
                    refCtr = tEpoch + 1
                refCtr -= 1
                    
    for gen in range(numSpikeGen):
        spikeTimes.append(np.where(spikes[gen, :])[0].tolist()) 
    return spikeTimes

# Produce spike times for the resetting spikes
def genResetSpikeTimes(runTime, numSpikeGen, numESubPop, repetition):
    runTimeRep = runTime//repetition
    spikeTimes = []
    initialIdleTime = 300
    spikes = np.zeros((numSpikeGen, runTime)) 
    for gen in range(numSpikeGen):
        flag_pre = 0
        flag = 0
        rep = 0
        for time in range(runTime):
            if time >= int(runTimeRep/numESubPop*(rep+2/3))+initialIdleTime \
                          and time < int(runTimeRep/numESubPop*(rep+1))+initialIdleTime:
                spikes[gen][time] = np.random.rand(1) < 0.6
                flag = 1
            else: 
                flag = 0
            if flag == 0 and flag_pre == 1:
                rep = rep + 1
            flag_pre = flag
    for gen in range(numSpikeGen):
        spikeTimes.append(np.where(spikes[gen, :])[0].tolist()) 
    return spikeTimes

# Produce spike times of the noise sources
def genNoiseSpikeTimes(runTime, spikingProbability, numSpikeGen):
    spikeTimes = []
    binSize = 300
    nbin = runTime//binSize
    for port in range(numSpikeGen):
        spikes_buffer = np.array([])
        for ii in range(nbin):
            spikes = np.random.rand(binSize) < spikingProbability
            spikes_buffer = np.append(spikes_buffer, spikes)
        spikeTimes.append(np.where(spikes_buffer)[0].tolist()) #[0] to extract row indices, [1] to extract column indices
    return spikeTimes

# Calculate the mean post-synaptic spiking frequency of the excitatory population over time for each subpopulation
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores):    
    numECorePerPop = 128
    numPop = numCores//numECorePerPop
    window_size = 100
    window = np.ones(window_size) # Window size of 25 time steps for binning.
    buffer_in_avg = [[] for ii in range(numPop)]
    for pop in range(numPop):
        buffer = np.zeros(())
        for neuron in range(numECorePerPop):
            data = postSynapticSpikes.data[pop*numECorePerPop+neuron]
            binned = np.asarray([np.convolve(data, window)])[:, :-window_size + 1]
            if neuron == 0:
                buffer = binned
            else:
                buffer = np.concatenate((buffer, binned), axis = 0)
        buffer_in_avg[pop] = buffer.mean(axis=0)       
    return buffer_in_avg                        
    
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
        load = "_30_4"
    elif (runTime == 30300):
        load = "_30" # === 2 pop good
    else:
        load = "lava"

    print(f"E-Neurons: {numECores}")
    print(f"Jee = {Jee}")
    print(f"learning: {learning}")
    print(f"ImpParam = {ImpParam}")
    print(f"x1Tau = {x1Tau}")
    print(f"y1Tau = {y1Tau}")
    print(f"tEpochlr = {tEpochlr}")
    # Generate spike times

    print("loading spikes from disk")
        
    spikeTimes = []
    with open("data/stdp_multi_retrial/spiketimes/spikeTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeTimes.append(sublist)

    spikeResetTimes = []
    with open("data/stdp_multi_retrial/spiketimes/spikeResetTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeResetTimes.append(sublist)

    spikeENoiseTimes = []
    with open("data/stdp_multi_retrial/spiketimes/spikeENoiseTimes" + load + ".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeENoiseTimes.append(sublist)

    spikeINoiseTimes = []
    with open("data/stdp_multi_retrial/spiketimes/spikeINoiseTimes" + load + ".txt", "r") as f:
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
        filename1 = "multi_ahm"

        cg1Spikes = []
        with open("data/stdp_multi_retrial/output_data/cg1Spikes_STDP"+filename1+".txt", "r") as f:
            for line in f:
                sublist = list(map(int, line.strip().split()))
                cg1Spikes.append(sublist)
        
        cg1_Voltage = np.load("data/stdp_multi_retrial/output_data/cg1_Voltage"+filename1+".npy")
        meanFreqOutArray = np.load("data/stdp_multi_retrial/output_data/meanFreqOutArray_STDP"+filename1+".npy")
        weightMatrix = np.load("data/stdp_multi_retrial/output_data/weightmatrix"+filename1+".npy")

        # Shape weight matrices 
        # weightMatrix = []
        weight_ave = [] #[[] for _ in range(numESubPop)] # 8 time lists
        weight_std = [] #[[] for _ in range(numESubPop)]
        sumWeight = np.zeros(runTime//dt)
        count = np.zeros(runTime//dt)
        
        for pop in range(numESubPop):
            weight_ave.append(np.nanmean(weightMatrix[pop*numESubCores:(pop+1)*numESubCores, \
                                                            pop*numESubCores:(pop+1)*numESubCores]))
            weight_std.append(np.nanstd(weightMatrix[pop*numESubCores:(pop+1)*numESubCores, \
                                                            pop*numESubCores:(pop+1)*numESubCores]))
        # print(weight_ave)          
        print("End of Runtime  | Avg | Std | Wgt. of Subpopulations")
        string = str(runTime) + " | "
        for pop in range(numESubPop):
            string = string + str(round(weight_ave[pop])) + " | "
            string = string + str(round(weight_std[pop])) + " | "
        print(str(string))


        #weight matrix
        import datetime
        filename1 = "multi_ahm_new"
        print(filename1)


        if (plotting):

            # # Plot probability distributions of the synaptic weights over time
            # fig = plt.figure(1002, figsize=(40, 40))   
            # for time in range(4):
            #     ax = plt.subplot(2, 2, time+1)
            #     array = np.ravel(weightMatrix[time][:numESubCores, :numESubCores])
            #     mask = ~np.isnan(array)
            #     array = array[mask]
            #     hist, bins = np.histogram(array, bins='auto', density=True)
            #     bin_centers = (bins[:-1] + bins[1:]) / 2
            #     # Plot the histogram
            #     plt.plot(bin_centers, hist)
            #     plt.xlabel('Efficacy', fontsize=60)
            #     plt.ylabel('Probability density', fontsize=60)
            #     plt.xticks(fontsize=45)
            #     plt.yticks(fontsize=45)
            #     #plt.ylim(0, 0.15)

            # if haveDisplay:
            #     plt.show()
            # else:
            #     fileName = "figures/multi_population_learning/SynapticMatrices_Distribution" + str(trial+1) + ".png"
            #     print("No display available, saving to file " + fileName + ".")
            #     fig.savefig(fileName)

            
            # Process and plot pre,  postsynaptic spikes, membrane voltage and excitatory frequency over time
            fig = plt.figure(1001, figsize=(100, 70))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']

            ax0 = plt.subplot(5, 1, 1)
            plt.eventplot(spikeTimes, color='black') #512
            #for i in range(len(spikeTimes)):
            #    plt.plot(spikeTimes[i], np.repeat(i,len(spikeTimes[i])), marker=".", markersize=0.1, color='black') #512
            #plt.title('presynaptic spikes', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30000,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=80)
            plt.yticks(fontsize=80)
            ax0.set_ylim([0,512])
            plt.ylabel("Neuron ID", fontsize = 80)


            cg1Spikes_raster = []
            for sublist in cg1Spikes:
                indexes = [i for i, value in enumerate(sublist) if value == 1]
                cg1Spikes_raster.append(indexes)



            ax1 = plt.subplot(5, 1, 2)
            # cg1Spikes[0].plot() #512
            plt.eventplot(cg1Spikes_raster, color='black')
            #for i in range(len(cg1Spikes_raster)):
            #    plt.plot(cg1Spikes_raster[i], np.repeat(i,len(cg1Spikes_raster[i])), marker=".", markersize=0.1, color='black') #512
            #plt.title('postsynaptic spikes', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30000,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=80)
            plt.yticks(fontsize=80)
            plt.xlabel("", fontsize = 80)
            plt.ylabel("Neuron ID", fontsize = 80)
            ax1.set_ylim([0,512])

            ax2 = plt.subplot(5, 1, 3)
            for pop in range(numECores//128):
                plt.plot(cg1_Voltage[pop])
                # v1 = cg1_Voltage[pop].plot() #4
            #plt.title('Membrane', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30000,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=16)
            #plt.yticks(fontsize=16)
            plt.yticks(ticks=np.linspace(0,15000,3), labels=["0", "7,500", "15,000"], fontsize=80)
            ax2.set_ylim([0,14000])
            plt.xlabel("", fontsize = 80)
            plt.ylabel("Membrane", fontsize = 80)

            ax0.set_xlim(0, runTime)
            ax1.set_xlim(ax0.get_xlim())
            ax2.set_xlim(ax0.get_xlim())
            #ax2.set_ylim([0,512])

            # if haveDisplay:
            #     plt.show()
            # else:
            #     fileName = "figures/multi_population_learning/PSTH" + str(trial+1) + ".png"
            #     print("No display available, saving to file " + fileName + ".")
            #     fig.savefig(fileName)
            

            ax4 = plt.subplot(5, 1, 4)
            for pop in range(numESubPop):
                ax4.plot(meanFreqOutArray[pop], color=colors[pop], label="subpop "+str(pop+1))
            #plt.xlabel("time", fontsize = 16)
            plt.ylabel("freq [/100 ts]", fontsize =80)
            #plt.title('spiking activity of excitatory population over time', fontsize = 16)
            #plt.xticks(fontsize=16)
            plt.xticks(ticks=np.linspace(0,30000,7), labels=["0", "5,000", "10,000", "15,000", "20,000", "25,000", "30,000"], fontsize=80)
            plt.yticks(fontsize=80)
            plt.legend(loc="best", fontsize=80)
            
            ax4.set_xlim(ax0.get_xlim())
            ax4.set_ylim([0,25])
            #ax4.axhline(y=12.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=13, color='black', linestyle='--', linewidth=5)
            #ax4.axhline(y=13.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=14, color='black', linestyle='--', linewidth=5)
            #ax4.axhline(y=14.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=15, color='black', linestyle='--', linewidth=5)
            #ax4.axhline(y=15.5, color='black', linestyle='--', linewidth=5)

            ax5 = plt.subplot(5, 1, 5)
            for pop in range(numESubPop):
                ax5.plot(meanFreqOutArray[pop], color=colors[pop])
            plt.xlabel("Time", fontsize = 80)
            plt.ylabel("Freq. [/100 ts]", fontsize =80)
            #plt.legend(loc="best", fontsize=80)
            
            #plt.title('spiking activity of excitatory population over time', fontsize = 16)
            plt.xticks(ticks=np.linspace(0,30000,16), labels=["0",  "2,000",  "4,000",  "6,000",  "8,000", "10,000", "12,000", "14,000", "16,000", "18,000", "20,000", "22,000", "24,000", "26,000", "28,000", "30,000"], fontsize=80)
            #plt.xticks(fontsize=16)
            plt.yticks(fontsize=80)

            ax5.set_xlim([24000, 30000])
            ax5.set_ylim([0,25])
            #ax4.axhline(y=12.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=13, color='black', linestyle='--', linewidth=5)
            ax5.axhline(y=13.5, color='black', linestyle='--', linewidth=2)
            # ax4.axhline(y=14, color='black', linestyle='--', linewidth=5)
            #ax4.axhline(y=14.5, color='black', linestyle='--', linewidth=5)
            # ax4.axhline(y=15, color='black', linestyle='--', linewidth=5)
            #ax4.axhline(y=15.5, color='black', linestyle='--', linewidth=5)




            #if haveDisplay:
            #    plt.show()
            #else:
            fileName = "figures/stdp/PSTH_plot" + str(trial+1) +  filename1 + ".png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)



if __name__ == "__main__":        
    main()
