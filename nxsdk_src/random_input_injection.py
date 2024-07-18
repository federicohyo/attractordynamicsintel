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
                 numEPerCore,
                 numICores, 
                 numIPerCore,
                 numSpikeGen, 
                 tau, 
                 E1pc,
                 Einh,
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
    cg1 = net.createCompartmentGroup(size=0) 
    Ecg = []
    for coreID in range(numECores//numEPerCore):
        compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=300,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3,
            enableSpikeBackprop=1,
            enableSpikeBackpropFromSelf=1,
            logicalCoreId = coreID,
            )
        Ecg.append(net.createCompartmentGroup(size=numEPerCore, prototype=compartmentPrototype1))
        cg1.addCompartments(Ecg[coreID])
   

    # -------------------------------------------------------------------------
    # Make an inhibitory population 
    # -------------------------------------------------------------------------
    #cg2... inhibitory population
    cg2 = net.createCompartmentGroup(size=0)
    for coreiID in range(numICores//numIPerCore):
        compartmentPrototype2 = nx.CompartmentPrototype(
                vThMant=300,         #vThMant*2^(6+vTHexp)
                functionalState=2,    #IDLE... Compartment gets serviced
                compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
                refractoryDelay = 3,
                logicalCoreId = numECores//numEPerCore+coreiID,
                )
        cg2.addCompartments(net.createCompartmentGroup(size=numIPerCore, prototype=compartmentPrototype2))
   
            
    # Create an E-STDP learning rule used by the learning-enabled synapse and connect the pre synaptic spike generator.
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-56)*x1*y0-2^-4*x1*y0+2^-4*sgn(w-56)*y1*x0+2^-4*y1*x0',
    lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-2^6)*x1*y0-2^-4*x1*y0',
                                x1Impulse=x1Imp,
                                x1TimeConstant=x1Tau,
                                y1Impulse=y1Imp,
                                y1TimeConstant=y1Tau,
                                tEpoch=1)  

    connProtoRecurrent = nx.ConnectionPrototype(weight=Jee,  #weight initialized at Jee
                                                enableLearning=1, 
                                                learningRule=lr,  
                                                delay=0,
                                                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                numWeightBits=8,
                                                weightLimitMant=7,
                                                weigthLimitExp=1,
                                                weightExponent=0
                                                ) 
    
    #Other connection prototypes
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=0)
    connProto3 = nx.ConnectionPrototype(weight=100, delay=0)
    connProto4 = nx.ConnectionPrototype(weight=200, delay=0) 
    
    #connect 10 E1pc to each excitatory subpopulation 
    synaptic_matrix0 = np.eye(numECores)
    E1pc.connect(cg1, prototype=connProto4, connectionMask=synaptic_matrix0) 
        
    #connect E1pc and the inhibitory population
    synaptic_matrix1 = np.random.rand(numICores, numSpikeGen) < 0.03
    E1pc.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix1) 
        
    #connect Einh and the inhibitory population (reset)
    Einh.connect(cg2, prototype=connProto4, connectionMask=np.ones((numICores, 32)))
   
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
    synaptic_matrix6 = np.random.rand(numECores, numECores) < synaptic_connectivity6 
    flag = 0
    flipped = np.zeros((numECores, numECores))
    for row in range(numECores):
        for col in range(numECores):
            if row == col:
                synaptic_matrix6[row][col] = 0
            if synaptic_matrix6[col][row] == 1:
                if flag and flipped[row][col]:
                    synaptic_matrix6[col][row] = 0
                    flipped[col][row] = 1
                    flag = 0
                elif flag and flipped[col][row]: 
                    synaptic_matrix6[row][col] = 0
                    flipped[row][col] = 1
                    flag = 1        
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
    wPc = nx.IntervalProbeCondition(dt=dt, tStart=dt-1)
    #sPc = nx.SpikeProbeCondition(dt=1, tStart=60000)
    weightProbe = [[] for _ in range(len(connectRecurrent))]
    SpikesProbe = cg1.probe(nx.ProbeParameter.SPIKE, probeConditions = None)
    for row in range(len(connectRecurrent)):
        for connection in range(len(connectRecurrent[row])):
            if connectRecurrent[row][connection]==0:
                # Fill no connections with 0
                weightProbe[row].append(0)
            else:
                weightProbe[row].append(connectRecurrent[row][connection].probe(nx.ProbeParameter.SYNAPSE_WEIGHT, probeConditions = wPc)) 
    return SpikesProbe, weightProbe


# Create numSpikeGen spike generators with spike timing, spikeTimes
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) 
    return spikeGen

def genStimulusSpikeTimes(spikeArray, 
                          target_neurons,
                          stimulus_number,
                          stimulation_interval,
                          reset_interval,
                          numResetGen,
                          spikingProbability, 
                          tEpoch):
    runTime = np.size(spikeArray, axis = 1)
    numSpikeGen = np.size(spikeArray, axis = 0)
    initialIdleTime = 300
    startTime = initialIdleTime+stimulus_number*(stimulation_interval+reset_interval)
    endTime = initialIdleTime+stimulus_number*(stimulation_interval+reset_interval) + stimulation_interval
    for neuron in target_neurons:
        for time in range(startTime, endTime):
            refCtr = 0
            if time < runTime:
                spikeArray[neuron, time] = (np.random.rand(1) < spikingProbability) and refCtr <= 0 
                if spikeArray[neuron, time]:
                    refCtr = tEpoch + 1
                refCtr -= 1
            else:
                print("Warning!! Spike time exceeded runTime. spike array not filled.")
                break

    spikeTimes = []
    for neuron in range(numSpikeGen):
        spikeTimes.append(np.where(spikeArray[neuron, :])[0].tolist()) 
    spikeResetArray = np.zeros((numResetGen, runTime))    
    startTime = int(endTime + reset_interval*0.8)
    endTime = endTime + reset_interval
    for gen in range(numResetGen):
        for time in range(startTime, endTime):
            spikeResetArray[gen, time] = 1
    spikeResetTimes = []
    for gen in range(numResetGen):
        spikeResetTimes.append(np.where(spikeResetArray[gen, :])[0].tolist()) 
    return spikeTimes, spikeResetTimes
    

# Calculate the mean frequency of output spikes from a probe of size = numCores for each sample period
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores, numNeuronsSubPop):    
    numPop = numCores//numNeuronsSubPop
    window_size = 50
    window = np.ones(window_size) # Window size of 25 time steps for binning.
    buffer_in_avg = [[] for ii in range(numPop)]
 
    for pop in range(numPop):
        buffer = np.zeros(())
        for neuron in range(numNeuronsSubPop):
            binned = (100//window_size)*np.asarray([np.convolve(postSynapticSpikes.data[pop*numNeuronsSubPop+neuron], \
            window)])[:, :-window_size + 1]
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
if __name__ == "__main__":        
        #------------------------------------------------------------------------
    # Create a network
    #------------------------------------------------------------------------  
    # If numECores = 256, the range of tSNE_out should be set within 16x16 matrix: [0 15]-[0 15]
    #     0   1   2  ...                                    15
    # 0   0   1   2  3   4  5  6  7   8  9  10 11  12 13 14 15
    # 1  16  17 18 19  20 21 22 23  24 25  26 27  28  29 30 31 
    # 2  ...
    # ...
    # 15
    
    # Divide this into 16 groups of 16 neurons
    # When target_neuron lands on one of these groups, the whole group is stimulated together
    # Number of excitatory neurons
    numECores = 256
    numNeuronsSubPop = 49 #odd integer which is also a square of an integer
    tSNE_out = []
    target_neurons = []
    print("Order of neurons in the matrix: ")
    print(np.arange(0, numECores).reshape(int(math.sqrt(numECores)), int(math.sqrt(numECores))))
    for sample in range(60):
        tSNE_out.append(np.array([random.randrange(4, 8, 1), \
                                  random.randrange(3, 12, 8)]))
        target_neuron = tSNE_out[sample][0] + tSNE_out[sample][1]*math.sqrt(numECores)
        print("target neuron:")
        print(target_neuron)
        side = math.sqrt(numNeuronsSubPop)

        start_left = - (side-1)//2
        end_right = (side-1)//2
        if target_neuron % math.sqrt(numECores) < (side-1)//2:
            start_left = - (target_neuron % math.sqrt(numECores))
        elif target_neuron % math.sqrt(numECores) >= math.sqrt(numECores) - (side-1)//2:
            end_right = (math.sqrt(numECores)-1) - target_neuron % math.sqrt(numECores) 

        offset = 0
        start_top = - (side-1)//2
        end_bottom = (side-1)//2
        if target_neuron // math.sqrt(numECores) < (side-1)//2:
            start_top = - (target_neuron // math.sqrt(numECores))
        elif target_neuron // math.sqrt(numECores) >= math.sqrt(numECores) - (side-1)//2:
            end_bottom = (math.sqrt(numECores)-1) - target_neuron // math.sqrt(numECores)

        target_neurons_buf = np.array([])
        for val in np.arange(start_top, end_bottom+1):
            start_neuron = target_neuron + start_left + val*math.sqrt(numECores) 
            end_neuron = target_neuron + end_right + val*math.sqrt(numECores) 
            target_neurons_buf = np.concatenate((target_neurons_buf, \
            np.arange(start_neuron, end_neuron+1)), axis=0).astype(int)

        target_neurons.append(target_neurons_buf)
    
    # Time duration of the execution
    runTime = 61300
    print("The total number of timesteps is ", runTime, ".")
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = 0.9
    noiseSpikingProbability = 0.02
    # Number of excitatory neurons per core
    numEPerCore = numECores//2
    # Number of inhibitory neurons
    numICores = numECores//2
    numIPerCore = numICores
    # Number of generators
    numSpikeGen = numECores
    numResetGen = 32
    # Time constant
    tau=20
    # Refractory period of the input spikes
    tEpoch = 1
    stimulation_interval = 500
    reset_interval = 500
    # Initial recurrent weight (learning)
    Jee = 0
    # Learning parameters
    x1Imp=16
    y1Imp=16
    x1Tau=4
    y1Tau=4
    
    # Sample cycle of weights
    dt = runTime//4
    
    # Initialize spikeArray with noiseSpikingProbability
    spikeArray = np.random.rand(numSpikeGen, runTime) < noiseSpikingProbability
    stimulus_number = 0
    for sample in range(len(target_neurons)):
        (spikeTimes_buffer, spikeTimesReset_buffer) = genStimulusSpikeTimes(spikeArray,
                                                                            target_neurons[sample], 
                                                                            stimulus_number,
                                                                            stimulation_interval,
                                                                            reset_interval,
                                                                            numResetGen,               
                                                                            inputSpikingProbability, 
                                                                            tEpoch)
        stimulus_number = stimulus_number + 1
        if sample == 0:
            spikeTimes = spikeTimes_buffer
            spikeTimesReset = spikeTimesReset_buffer
            spikeArray = np.zeros((numSpikeGen, runTime))
        else:
            for gen in range(numSpikeGen):
                spikeTimes[gen] = spikeTimes[gen]+spikeTimes_buffer[gen]
                if gen < numResetGen:
                    spikeTimesReset[gen] = spikeTimesReset[gen] + spikeTimesReset_buffer[gen]
            
    net = nx.NxNet()        
    Epc1 = createSpikeGenerators(net, numSpikeGen, spikeTimes) 
    Einh = createSpikeGenerators(net, numResetGen, spikeTimesReset)
    #--------------------------------------------------------------------
    # Configure network
    #--------------------------------------------------------------------   
    print("Spikes were generated. Configuring network...")    
    (cg1Spikes, weightProbe) = setupNetwork(net, 
                                          numECores,
                                          numEPerCore,
                                          numICores,  
                                          numIPerCore,
                                          numSpikeGen, 
                                          tau,           
                                          Epc1, 
                                          Einh,
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

    net.run(runTime, executionOptions={'partition':'nahuku32_2h'})
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
                    weightArray[time] = np.append(weightArray[time], 0)
            else: 
                for time in range(runTime//dt):
                    weightArray[time] = np.append(weightArray[time], [weightProbe[row][connection][0].data[time]])
      
    # Shape weight matrices 
    weightMatrix = []
    for time in range(len(weightArray)):
        # Reshape the weight vectors into matrices
        weightMatrix.append(weightArray[time].reshape(numECores,numECores))

        
    fig2 = plt.figure(1003, figsize=(40, 40))    
    cmap = colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'black'])
    for time in range(len(weightMatrix)):
        ax = plt.subplot(2, 2, time+1)
        # Create a heatmap using matplotlib
        plt.imshow(weightMatrix[time], cmap=cmap, interpolation='nearest', vmin=0, vmax=64)
        cbar = plt.colorbar(shrink=0.9)
        cbar.ax.tick_params(labelsize=45)
        plt.title("t="+str((time+1)*dt), fontsize=60)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/SynapticMatrices_tSNE.png"
        print("No display available, saving to file " + fileName + ".")
        fig2.savefig(fileName)
    #------------------------------------------------------------------------
    # Process and plot pre,  postsynaptic spikes, membrane voltage and excitatory frequency over time
    #------------------------------------------------------------------------    
    fig = plt.figure(1001, figsize=(100, 40))
    
    ax0 = plt.subplot(2, 1, 1)
    sp_in = plotRaster(spikeTimes)
    plt.title('presynaptic spikes', fontsize = 100)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.ylabel("probe index", fontsize = 80)
    
    ax1 = plt.subplot(2, 1, 2)
    sp_out = cg1Spikes[0].plot()
    plt.title('postsynaptic spikes', fontsize = 100)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.xlabel("", fontsize = 60)
    plt.ylabel("probe index", fontsize = 80)

    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/Attractor_Dynamics_tSNE.png"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)
        
    print("Saving data...")    
    postSpikeTimes = extractPostSynapticSpikes(runTime, cg1Spikes[0], numECores)
    data = [target_neurons, postSpikeTimes, weightMatrix]    
    with open('data/data_tSNE.pkl', 'wb') as f:
        pickle.dump(data , f)
