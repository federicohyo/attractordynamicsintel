# Import modules

import matplotlib.pyplot as plt
# import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
# from nxsdk.utils.plotutils import plotRaster
import random
import string
random.seed(42)
# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Calculate the mean post-synaptic spiking frequency of the excitatory population
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores, windowsize=100):    
    numECorePerPop = numCores
    numPop = numCores//numECorePerPop
    window_size = windowsize
    window = np.ones(window_size) # Window size of 25 time steps for binning.
    buffer_in_avg = [[] for ii in range(numPop)]
 
    for pop in range(numPop):
        buffer = np.zeros(())
        for neuron in range(numECorePerPop):
            binned = (100//window_size)*np.asarray([np.convolve(postSynapticSpikes[pop*numECorePerPop+neuron], \
            window)])[:, :-window_size + 1]
            if neuron == 0:
                buffer = binned
            else:
                buffer = np.concatenate((buffer, binned), axis = 0)
        buffer_in_avg[pop] = buffer.mean(axis=0)       
    return buffer_in_avg     

def main(): 
    #Time duration of the execution
    runTime = 1500
    print("The total number of timesteps is ", runTime, ".")
    
    filename1 = "final2"

    #IMPORT the data to reproduce the graphs
    spikeTimes = []
    with open("self_sustained_ok/spiketimes_inp_self_sustained"+filename1+".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            spikeTimes.append(sublist)    

    spikeTimes_scatter = []
    for sublist in spikeTimes:
        row = [1 if i in sublist else 0 for i in range(1,1500)]
        spikeTimes_scatter.append(row)
    
    cg1Spikes = []
    with open("self_sustained_ok/cg1Spikes_self_sustained"+filename1+".txt", "r") as f:
        for line in f:
            sublist = list(map(int, line.strip().split()))
            cg1Spikes.append(sublist)   
    # print(type(cg1Spikes))
    cg1Spikes_raster = []
    for sublist in cg1Spikes:
        indexes = [i for i, value in enumerate(sublist) if value == 1]
        cg1Spikes_raster.append(indexes)

    neuron = 69
    cg1_0Voltage = np.load("self_sustained_ok/voltages/cg1_"+str(neuron)+"Voltage_self_sustained"+filename1+".npy")
    cg1_0VoltageIn = np.load("self_sustained_ok/cg1_0VoltageIn_self_sustained"+filename1+".npy")
    # meanFreqOutArray1 = np.load("self_sustained_ok/meanFreqOutArray_self_sustained"+filename1+".npy")
    print(np.array(cg1Spikes).shape)
    meanFreqOutArray = calculateMeanFreqOut(1500, np.array(cg1Spikes), 128, 50)[0]
    print(meanFreqOutArray)
    # print(meanFreqOutArray1)

    size = 15
    size2 = 10
    ticks = 35
    # PLOT
    fig = plt.figure(1001, figsize=(35,25))

    ax0 = plt.subplot(4, 1, 1)
    plt.eventplot(spikeTimes, orientation='horizontal', linelengths=1, colors='black', rasterized = True)
    plt.title('Presynaptic spikes', fontsize = 30+size)
    # plt.xticks(fontsize=20)
    plt.yticks(fontsize=ticks)
    plt.ylabel("Neuron ID", fontsize = 30+size2)
    ax0.set_ylim([0,127])
    plt.xticks(ticks=np.linspace(0,1500,6), labels=["0", "300", "600", "900", "1,200", "1,500"], fontsize=ticks)
    ax0.text(0, 1.02, string.ascii_uppercase[0], transform=ax0.transAxes, size=40, weight='bold')

    ax1 = plt.subplot(4, 1, 2)
    plt.eventplot(cg1Spikes_raster, orientation='horizontal', linelengths=1, colors='black', rasterized = True)
    plt.title('Postsynaptic spikes', fontsize = 30+size)
    # plt.xticks(fontsize=40)
    plt.yticks(fontsize=ticks)
    plt.xlabel("", fontsize = 30+size2)
    plt.ylabel("Neuron ID", fontsize = 30+size2)
    ax1.set_ylim([0,127])
    plt.xticks(ticks=np.linspace(0,1500,6), labels=["0", "300", "600", "900", "1,200", "1,500"], fontsize=ticks)
    ax1.text(0, 1.02, string.ascii_uppercase[1], transform=ax1.transAxes, size=40, weight='bold')

    ax2 = plt.subplot(4, 1, 3)
    plt.plot(meanFreqOutArray, color='black')
    plt.xlabel("", fontsize = 30+size2)
    plt.ylabel("Freq. [/100 timesteps]", fontsize = 30+size2)
    plt.title('Spiking activity of excitatory population over time', fontsize = 30+size)
    # plt.xticks(fontsize=20)
    plt.yticks(fontsize=ticks)
    # freq_avg = sum(meanFreqOutArray[2*(runTime//3):runTime])/len(meanFreqOutArray[2*(runTime//3):runTime])
    # freq_avg = np.mean(meanFreqOutArray[2*(runTime//3):runTime])
    # print(freq_avg)
    ax2.axhline(y=13, color='black', linestyle='--', linewidth=5)
    plt.xticks(ticks=np.linspace(0,1500,6), labels=["0", "300", "600", "900", "1,200", "1,500"], fontsize=ticks)
    ax2.text(0, 1.02, string.ascii_uppercase[2], transform=ax2.transAxes, size=40, weight='bold')

    ax3 = plt.subplot(4, 1, 4)
    plt.plot(cg1_0Voltage, color='black')
    plt.title('Membrane voltage', fontsize = 30+size)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=ticks)
    plt.xlabel("Timesteps", fontsize = 30+size2)
    plt.ylabel("Membrane voltage", fontsize = 30+size2)
    ax3.axhline(y=max(cg1_0Voltage), color='black', linestyle='--', linewidth=5)
    plt.xticks(ticks=np.linspace(0,1500,6), labels=["0", "300", "600", "900", "1,200", "1,500"], fontsize=ticks)
    plt.yticks(ticks=np.linspace(0,12000,7), labels=["0", "2,000", "4,000", "6,000", "8,000", "10,000", "12,000"], fontsize=ticks)
    ax3.text(0, 1.02, string.ascii_uppercase[3], transform=ax3.transAxes, size=40, weight='bold')

    ax0.set_xlim(0, runTime)
    ax1.set_xlim(ax0.get_xlim())
    ax2.set_xlim(ax0.get_xlim())
    ax3.set_xlim(ax0.get_xlim())
    plt.tight_layout()

    # Save the figure
    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/self_sustained_activity.pdf"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)

if __name__ == "__main__":        
    main()
