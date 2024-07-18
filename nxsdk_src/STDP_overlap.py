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

haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Nx API

# plt is used for graphical displays

# -----------------------------------------------------------------------------
# Function: setupNetwork
# This function defines an example NxNet network, uses N2Compiler to compile
# and initialize the software mappings to hardware components and
# finally returns the board
# -----------------------------------------------------------------------------

def setupNetwork(net, 
                 numECores, 
                 numICores, 
                 numSpikeGen, 
                 numESubPop,
                 tau, 
                 Epc1, 
                 Ireset,
                 Enoise,
                 Inoise,
                 Jee, 
                 tEpoch, 
                 x1Imp, 
                 y1Imp,
                 x1Tau,
                 y1Tau,
                 dt):
    # -------------------------------------------------------------------------
    # Create numECores compartments and make a compartment group
    # Make an excitatory population
    # -------------------------------------------------------------------------
    #cg1... excitatory population
    # The noise term is calculated by the following equation:
    # (lfsr() - 2**7 + noiseMantAtCompartment * 2**6) * 2**(noiseExpAtCompartment-7)
    cg1 = []
    numEperCore = 128
    cg1 = net.createCompartmentGroup(size=0) 
    Ecg = []
    for coreID in range(numECores//numEperCore):
        compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3,
            enableSpikeBackprop=1,
            enableSpikeBackpropFromSelf=1,
            logicalCoreId = coreID,
            )
        Ecg.append(net.createCompartmentGroup(size=numEperCore, prototype=compartmentPrototype1))
        cg1.addCompartments(Ecg[coreID])
   

    # -------------------------------------------------------------------------
    # Make an inhibitory population 
    # -------------------------------------------------------------------------
    #cg2... inhibitory population
    numIperCore = 64
    cg2 = net.createCompartmentGroup(size=0)
    for coreiID in range(numICores//numIperCore):
        compartmentPrototype2 = nx.CompartmentPrototype(
                vThMant=180,         #vThMant*2^(6+vTHexp)
                functionalState=2,    #IDLE... Compartment gets serviced
                compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
                refractoryDelay = 3,
                logicalCoreId = numECores//numEperCore+coreiID,
                )
        cg2.addCompartments(net.createCompartmentGroup(size=numIperCore, prototype=compartmentPrototype2))
   
            
    # Create an E-STDP learning rule used by the learning-enabled synapse and connect the pre synaptic spike generator.
    lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-86)*x1*y0-2^-4*x1*y0',
                                x1Impulse=x1Imp,
                                x1TimeConstant=x1Tau,
                                y1Impulse=y1Imp,
                                y1TimeConstant=y1Tau,
                                tEpoch=2)  

    connProtoRecurrent = nx.ConnectionPrototype(weight=Jee,  #weight initialized at Jee
                                                enableLearning=1, 
                                                learningRule=lr,  
                                                delay=0,
                                                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                numWeightBits=8,
                                                weightLimitMant=2,
                                                weigthLimitExp=2,
                                                ) 
    
    #Other connection prototypes
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=0)
    connProtoIn = nx.ConnectionPrototype(weight=180, delay=0)
    connProtoIn2 = nx.ConnectionPrototype(weight=90, delay=0)
    connProtoReset = nx.ConnectionPrototype(weight=60, delay=0)
    connProtoNoise = nx.ConnectionPrototype(weight=30, delay=0) 
    #connect each generator to one excitatory neuron
    synaptic_matrix0 = np.eye(numECores)
    Epc1.connect(cg1, prototype=connProtoIn, connectionMask=synaptic_matrix0) 
        
    #connect generators to inhibitory pool
    #synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
    synaptic_matrix1 = np.random.rand(numICores, numECores) < 0.01
    Epc1.connect(cg2, prototype=connProtoIn2, connectionMask=synaptic_matrix1) 
        
    #connect Ireset and the inhibitory population (reset)
    Ireset.connect(cg2, prototype=connProtoReset)
    
    #coonect noise source to neurons
    Enoise.connect(cg1, prototype=connProtoNoise, connectionMask=np.eye(numECores))
    Inoise.connect(cg2, prototype=connProtoNoise, connectionMask=np.eye(numICores))
   
    #connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.25
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3)     
    #recurrent connection of the inhibitory population
    synaptic_connectivity4 = 0.52
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    for row in range(numICores):
        for col in range(numICores):
            if row == col:
                synaptic_matrix4[row][col] = 0
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    #connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.25
    synaptic_matrix5 = np.zeros((numECores, numICores)) < synaptic_connectivity5
    for row in range(numECores):
        row_matrix = np.random.rand(1,numICores) < synaptic_connectivity5
        synaptic_matrix5[row][:] = row_matrix
    cg2.connect(cg1, prototype=connProto2, connectionMask=synaptic_matrix5)          
    
    #connect recurrent stimulus and the excitatory population
    synaptic_connectivity6 = 0.25 + 0.001 #0.001 to compensate for the bits that are to be flipped in loops
    synaptic_matrix6 = np.bitwise_and(np.random.rand(numECores, numECores) < synaptic_connectivity6, \
                                      np.invert(np.eye(numECores, dtype=bool))*1)
    connectRecurrent = []
    for row in range(numECores):
        rowConnection = []  
        for col in range(numECores):   
            if synaptic_matrix6[row][col]:    
                rowConnection.append(cg1[col].connect(cg1[row], prototype=connProtoRecurrent))      
            else: 
                # Fill no connections with 0
                rowConnection.append(0)
        connectRecurrent.append(rowConnection.copy()) 
    
    # -------------------------------------------------------------------------
    # Configure probes
    # -------------------------------------------------------------------------
    vPc = nx.IntervalProbeCondition(dt=dt, tStart=dt-1)
    weightProbe = [[] for _ in range(len(connectRecurrent))]
    SpikesProbe = cg1.probe(nx.ProbeParameter.SPIKE, probeConditions = None)
    VoltageProbe=[]
    for pop in range(numECores//numEperCore):
        VoltageProbe.append(cg1[pop*numEperCore].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None))
    for row in range(len(connectRecurrent)):
        for connection in range(len(connectRecurrent[row])):
            if connectRecurrent[row][connection]==0:
                # Fill no connections with 0
                weightProbe[row].append(0)
            else:
                weightProbe[row].append(connectRecurrent[row][connection].probe(nx.ProbeParameter.SYNAPSE_WEIGHT, probeConditions = vPc)) 
    return SpikesProbe, VoltageProbe, weightProbe


# Create numSpikeGen spike generators with spike timing, spikeTimes
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) 
    return spikeGen

# Produce spike timing of numSpikeGen spike generators
def genInputSpikeTimes(runTime, spikingProbability, numSpikeGen, numESubPop, repetition, random_shift):
    """Generates an approximate Poisson spike train with a refractory period of tEpoch after each spike in order to avoid multiple spikes within the same learning epoch."""
    runTimeRep = runTime//repetition
    spikeTimes = []
    initialIdleTime = 300
    spikes = np.zeros((numSpikeGen, runTime)) 
    numNeuronsPerSubpop = round(numSpikeGen/numESubPop)
    shift1 = [random.randint(random_shift[0][0], random_shift[0][1]) for _ in range(repetition)]
    print(shift1)
    shift2 = [random.randint(random_shift[1][0], random_shift[1][1]) for _ in range(repetition)]
    print(shift2)
    # Add input and noise to the matrix, spikes
    for pop in range(numESubPop):
        for gen in range(int(numNeuronsPerSubpop*pop), int(numNeuronsPerSubpop*(pop+1))):
            rep = 0
            for time in range(runTime):
                if numESubPop == 4:
                    if pop == 1:   
                        gen_shifted = gen - shift1[rep]
                    elif pop == 3:
                        gen_shifted = gen - shift2[rep]
                    else:
                        gen_shifted = gen
                else:
                    print("Overlap generation is hardcoded for 4 attractors!! Overlap is not applied.")
                    gen_shifted = gen
                if time >= int(runTimeRep*(rep+pop/numESubPop))+initialIdleTime \
                              and time < int(runTimeRep*(rep+(pop+1/3)/numESubPop))+initialIdleTime:
                    spikes[gen_shifted][time] = (np.random.rand(1) < spikingProbability[1])
                else:
                    spikes[gen][time] = (np.random.rand(1) < spikingProbability[0])
                    if time == int(runTimeRep*(rep+1))+initialIdleTime:  
                        rep += 1
                        
    for gen in range(numSpikeGen):
        spikeTimes.append(np.where(spikes[gen, :])[0].tolist()) 
    return spikeTimes

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

# Produce spike timing of numSpikeGen spike generators
def genNoiseSpikeTimes(runTime, spikingProbability, numSpikeGen):
    spikeTimes = []
    binSize = 100
    nbin = runTime//binSize
    for port in range(numSpikeGen):
        spikes_buffer = np.array([])
        for ii in range(nbin):
            spikes = np.random.rand(binSize) < spikingProbability
            spikes_buffer = np.append(spikes_buffer, spikes)
        spikeTimes.append(np.where(spikes_buffer)[0].tolist()) #[0] to extract row indices, [1] to extract column indices
    return spikeTimes

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
    
def extractPostSynapticSpikes(runTime, postSynapticSpikes, numCores):               
    spikeTimes = []
 
    for neuron in range(numCores):
        spikeTimes.append(postSynapticSpikes.data[neuron])
    return spikeTimes                    
    
# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main():
    
    # Time duration of the execution
    runTime = 120300
    print("The total number of timesteps is ", runTime, ".")
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = [0.005, 0.8] # [low frequency input, high frequency input]
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.5
    inputFraction = 3
    # Number of neurons in each excitatory subpopulation
    numESubCores = 128
    # Number of excitatory subpopulations
    numESubPop = 4
    # Number of excitatory neurons
    numECores = numESubCores*numESubPop
    # Number of inhibitory neurons
    numICores = numECores//2
    # Number of pixels
    numSpikeGen = numECores
    # Time constant
    tau=16
    # Refractory period of the input spikes
    tEpoch = 1
    repetition = (runTime-300)//6000 
    # Initial recurrent weight (learning)
    Jee = 5
    # Learning parameters
    ImpParam = [6]
    x1Tau=4
    y1Tau=4
    
    # Sample cycle of weights
    dt = runTime//4
    
    # range of random shift for 2 subpopulations
    random_shift = [[5, 5], [10, 10]]
    #--------------------------------------------------------------------
    # Generate spikes or load spikes from the last execution
    #--------------------------------------------------------------------   
    spikeTimes = genInputSpikeTimes(runTime, 
                                      inputSpikingProbability, 
                                      numSpikeGen, 
                                      numESubPop, 
                                      repetition,
                                      random_shift)
    spikeResetTimes = genResetSpikeTimes(runTime, 32, numESubPop, repetition)
    spikeENoiseTimes = genNoiseSpikeTimes(runTime, EnoiseSpikingProbability, numECores)
    spikeINoiseTimes = genNoiseSpikeTimes(runTime, InoiseSpikingProbability, numICores)   
        
    #--------------------------------------------------------------------
    # Configure network
    #--------------------------------------------------------------------   
    for trial in range(len(ImpParam)):
        net = nx.NxNet()
        x1Imp=ImpParam[trial]
        y1Imp=ImpParam[trial]
        
        Epc1 = createSpikeGenerators(net, numSpikeGen, spikeTimes) 
        Ireset = createSpikeGenerators(net, 32, spikeResetTimes)
        Enoise = createSpikeGenerators(net, numSpikeGen, spikeENoiseTimes) 
        Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes) 
        print("Spikes were generated. Configuring network...")    
        (cg1Spikes, cg1_0Voltage, weightProbe) = setupNetwork(net, 
                                                              numECores,
                                                              numICores,     
                                                              numSpikeGen, 
                                                              numESubPop,
                                                              tau,           
                                                              Epc1, 
                                                              Ireset,
                                                              Enoise,
                                                              Inoise,
                                                              Jee,            
                                                              tEpoch,             
                                                              x1Imp, 
                                                              y1Imp,
                                                              x1Tau,
                                                              y1Tau,
                                                              dt)

        #------------------------------------------------------------------------
        # Run    
        #------------------------------------------------------------------------

        net.run(runTime, executionOptions={'partition':'nahuku08'})
        net.disconnect()
        print("Execution finished. Plotting synaptic matrices...")

        #------------------------------------------------------------------------
        # Process and plot weights
        #------------------------------------------------------------------------

        # Extract weightProbe and unpack it into vectors 
        weightArray = [np.array([]) for _ in range(runTime//dt)]   
        for row in range(len(weightProbe)):
            for connection in range(len(weightProbe[row])):
                if weightProbe[row][connection]==0:
                    for time in range(runTime//dt):
                        weightArray[time] = np.append(weightArray[time], np.nan)
                else: 
                    for time in range(runTime//dt):
                        weightArray[time] = np.append(weightArray[time], [weightProbe[row][connection][0].data[time]])

        # Shape weight matrices 
        weightMatrix = []
        weight_ave = [[] for _ in range(len(weightArray))]
        sumWeight = np.zeros(runTime//dt)
        count = np.zeros(runTime//dt)

        for time in range(len(weightArray)):
            # Reshape the weight vectors into matrices
            weightMatrix.append(weightArray[time].reshape(numECores,numECores))
            for pop in range(numESubPop):
                weight_ave[time].append(np.nanmean(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))
        print("Time  | Avg Wgt. of Subpopulations")
        for time in range(len(weightArray)):
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
            plt.imshow(weightMatrix[time], cmap=cmap, interpolation='nearest', vmin=0, vmax=80)
            cbar = plt.colorbar(shrink=0.9)
            cbar.ax.tick_params(labelsize=45)
            plt.title("t="+str((time+1)*dt), fontsize=60)
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)

        if haveDisplay:
            plt.show()
        else:
            fileName = "figures/overlap/SynapticMatrices_Overlap" + str(trial) + ".png"
            print("No display available, saving to file " + fileName + ".")
            fig2.savefig(fileName)
        #-------------------------------------------------------------------------     
        # Plot probability distribution
        #-------------------------------------------------------------------------   
        fig = plt.figure(1002, figsize=(40, 40))   
        for time in range(4):
            ax = plt.subplot(2, 2, time+1)
            array = np.ravel(weightMatrix[time][:numESubCores, :numESubCores])
            mask = ~np.isnan(array)
            array = array[mask]
            hist, bins = np.histogram(array, bins='auto', density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Plot the histogram
            plt.plot(bin_centers, hist)
            plt.xlabel('Efficacy', fontsize=60)
            plt.ylabel('Probability density', fontsize=60)
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)
            #plt.ylim(0, 0.15)

        if haveDisplay:
            plt.show()
        else:
            fileName = "figures/overlap/SynapticMatrices_Distribution_Overlap" + str(trial) + ".png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)

        #------------------------------------------------------------------------
        # Process and plot pre,  postsynaptic spikes, membrane voltage and excitatory frequency over time
        #------------------------------------------------------------------------    
        fig = plt.figure(1001, figsize=(100, 70))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']

        ax0 = plt.subplot(4, 1, 1)
        sp_in = plotRaster(spikeTimes)
        plt.title('presynaptic spikes', fontsize = 100)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylabel("neuron index", fontsize = 80)

        ax1 = plt.subplot(4, 1, 2)
        sp_out = cg1Spikes[0].plot()
        plt.title('postsynaptic spikes', fontsize = 100)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        plt.xlabel("", fontsize = 60)
        plt.ylabel("neuron index", fontsize = 80)

        ax2 = plt.subplot(4, 1, 3)
        for pop in range(numECores//128):
            v1 = cg1_0Voltage[pop][0].plot()
        plt.title('membrane voltage', fontsize = 100)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        plt.xlabel("", fontsize = 60)
        plt.ylabel("voltage", fontsize = 80)

        ax0.set_xlim(0, runTime)
        ax1.set_xlim(ax0.get_xlim())
        ax2.set_xlim(ax0.get_xlim())

        if haveDisplay:
            plt.show()
        else:
            fileName = "figures/overlap/PSTH_Overlap" + str(trial) + ".png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)

        ax4 = plt.subplot(4, 1, 4)
        print("Calculating mean postsynaptic frequencies...")    
        meanFreqOutArray = calculateMeanFreqOut(runTime, cg1Spikes[0], numECores) #For runTime=40000, this takes 5 minutes
        for pop in range(numESubPop):
            ax4.plot(meanFreqOutArray[pop], color=colors[pop])
        plt.xlabel("time", fontsize = 80)
        plt.ylabel("frequency [/100 time steps]", fontsize =80)
        plt.title('spiking activity of excitatory population over time', fontsize = 100)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)

        ax4.set_xlim(ax0.get_xlim())

        if haveDisplay:
            plt.show()
        else:
            fileName = "figures/overlap/PSTH_Overlap" + str(trial) + ".png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)

        for ii in range(repetition//2):
            ax0.set_xlim(runTime*2*ii//repetition, runTime*2*(ii+1)//repetition)
            ax1.set_xlim(runTime*2*ii//repetition, runTime*2*(ii+1)//repetition)
            ax2.set_xlim(runTime*2*ii//repetition, runTime*2*(ii+1)//repetition)
            ax4.set_xlim(runTime*2*ii//repetition, runTime*2*(ii+1)//repetition)
            if haveDisplay:
                plt.show()
            else:
                fileName = "figures/overlap/PSTH_Zoom"+str(ii)+ "_Overlap" + str(trial) + ".png"
                print("No display available, saving to file " + fileName + ".")
                fig.savefig(fileName)

        with open('data/data_weight_multi_Overlap' + str(trial) + '.pkl', 'wb') as f:
            pickle.dump(weightMatrix , f) 
        print("Weights were saved.")
    
if __name__ == "__main__":        
    main()