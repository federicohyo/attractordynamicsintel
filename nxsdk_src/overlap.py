# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
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


def generateTarget(numECores, numNeuronsSubPop, sampleSize, numOverlapedPixels, probOverlap):
    if numNeuronsSubPop > numECores:
        print("Error: Subpopulation is too large.")
        return 0
    data_in = []
    target_neurons = []
    count_overlap = 0
    count_no_overlap = 0
    for sample in range(sampleSize):
        if np.random.rand(1) <= 0.5:
            target_neurons.append(np.arange(0, numNeuronsSubPop))
        else:
            if np.random.rand(1) < probOverlap:
                target_neurons.append(np.arange(numNeuronsSubPop-numOverlapedPixels, \
                                                 2*numNeuronsSubPop-numOverlapedPixels)) # Overlap
                count_overlap += 1
            else: 
                target_neurons.append(np.arange(numNeuronsSubPop, 2*numNeuronsSubPop)) # No overlap
                count_no_overlap += 1
    if numOverlapedPixels != 0:
        print("Overlap Probability is ", round(100*count_overlap/(count_overlap+count_no_overlap)), "%.")
    return target_neurons

'''def generateTarget(numECores, numNeuronsSubPop, sampleSize, numOverlapedPixels, probOverlap, overlap_steps):
    data_in = []
    target_neurons = []
    count = 0
     offset = math.sqrt(numNeuronsSubPop) + (numNeuronsSubPop-1)//2
    for sample in range(sampleSize):
        if np.random.rand(1) <= 0.5:
            data_in.append(np.array([(numNeuronsSubPop-1)//2 + 1, (numNeuronsSubPop-1)//2 + 1]))
        else:
            if np.random.rand(1) <= probOverlap:
                data_in.append(np.array([offset-numOverlapedPixels, offset-overlap_steps])) # Overlap
                count = 1
            else: 
                data_in.append(np.array([offset, offset-1])) # No overlap
        target_neuron = data_in[sample][0] + data_in[sample][1]*math.sqrt(numECores)
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
    return target_neurons'''

def setupNetwork(net, 
                 numECores, 
                 numEPerCore,
                 numICores, 
                 numIPerCore,
                 numSpikeGen, 
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
    cg1 = net.createCompartmentGroup(size=0) 
    Ecg = []
    for coreID in range(numECores//numEPerCore):
        compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
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
                vThMant=180,         #vThMant*2^(6+vTHexp)
                functionalState=2,    #IDLE... Compartment gets serviced
                compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
                refractoryDelay = 3,
                logicalCoreId = numECores//numEPerCore+coreiID,
                )
        cg2.addCompartments(net.createCompartmentGroup(size=numIPerCore, prototype=compartmentPrototype2))
   
            
    # Create an E-STDP learning rule used by the learning-enabled synapse and connect the pre synaptic spike generator.
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-56)*x1*y0-2^-4*x1*y0+2^-4*sgn(w-56)*y1*x0+2^-4*y1*x0',
    lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-80)*x1*y0-2^-4*x1*y0',
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
                                                weightLimitMant=3,
                                                weigthLimitExp=3,
                                                weightExponent=0
                                                ) 
    
    #Other connection prototypes
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=0)
    connProtoIn = nx.ConnectionPrototype(weight=60, delay=0)
    connProtoReset = nx.ConnectionPrototype(weight=60, delay=0)
    connProtoNoise = nx.ConnectionPrototype(weight=30, delay=0) 
    
    #connect each generator to one excitatory neuron
    synaptic_matrix0 = np.eye(numECores)
    E1pc.connect(cg1, prototype=connProtoIn, connectionMask=synaptic_matrix0) 
        
    #connect E1pc and the inhibitory population
    synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
    E1pc.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix1) 
        
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
              
# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
if __name__ == "__main__":          
    # Time duration of the execution
    runTime = 40300
    print("The total number of timesteps is ", runTime, ".")
    
    numECores = 256
    numNeuronsSubPop = 128 #odd integer which is also a square of an integer
    sampleSize = (runTime -300)//1000
    numOverlapedPixels = 8
    probOverlap = 1
    print("The number of overlapped pixels is", numOverlapedPixels)
    
    target_neurons = generateTarget(numECores, 
                                    numNeuronsSubPop, 
                                    sampleSize, 
                                    numOverlapedPixels, 
                                    probOverlap)    
    
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = 0.6
    EnoiseSpikingProbability = 0.2
    InoiseSpikingProbability = 0.4
    # Number of excitatory neurons per core
    numEPerCore = numECores//2
    # Number of inhibitory neurons
    numICores = numECores//2
    numIPerCore = numICores
    # Number of generators
    numSpikeGen = numECores
    numResetGen = 32
    # Time constant
    tau=16
    # Refractory period of the input spikes
    tEpoch = 1
    stimulation_interval = 400
    reset_interval = 600
    # Initial recurrent weight (learning)
    Jee = 0
    # Learning parameters
    x1Imp=20
    y1Imp=20
    x1Tau=4
    y1Tau=4
    
    # Sample cycle of weights
    dt = runTime//4
    
    # Initialize spikeArray with noiseSpikingProbability
    spikeArray = np.random.rand(numSpikeGen, runTime) < noiseSpikingProbability
    for stimulus_number in range(len(target_neurons)):
        (spikeTimes_buffer, spikeTimesReset_buffer) = genStimulusSpikeTimes(spikeArray,
                                                                            target_neurons[stimulus_number], 
                                                                            stimulus_number,
                                                                            stimulation_interval,
                                                                            reset_interval,
                                                                            numResetGen,               
                                                                            inputSpikingProbability, 
                                                                            tEpoch)
        if stimulus_number == 0:
            spikeTimes = spikeTimes_buffer
            spikeTimesReset = spikeTimesReset_buffer
            spikeArray = np.zeros((numSpikeGen, runTime))
        else:
            for gen in range(numSpikeGen):
                spikeTimes[gen] = spikeTimes[gen]+spikeTimes_buffer[gen]
                if gen < numResetGen:
                    spikeTimesReset[gen] = spikeTimesReset[gen] + spikeTimesReset_buffer[gen]
    
    spikeENoiseTimes = genNoiseSpikeTimes(runTime, EnoiseSpikingProbability, numECores)
    spikeINoiseTimes = genNoiseSpikeTimes(runTime, InoiseSpikingProbability, numICores)
    net = nx.NxNet()        
    Epc1 = createSpikeGenerators(net, numSpikeGen, spikeTimes) 
    Ireset = createSpikeGenerators(net, numResetGen, spikeTimesReset)
    Enoise = createSpikeGenerators(net, numSpikeGen, spikeENoiseTimes) 
    Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes) 
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
        plt.imshow(weightMatrix[time], cmap=cmap, interpolation='nearest', vmin=0, vmax=80)
        cbar = plt.colorbar(shrink=0.9)
        cbar.ax.tick_params(labelsize=45)
        plt.title("t="+str((time+1)*dt), fontsize=60)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/overlap/SynapticMatrices_overlap" + str(numOverlapedPixels) + ".png"
        fig2.savefig(fileName)
        print("No display available, saved to file " + fileName + ".")
    #------------------------------------------------------------------------
    # Process and plot pre,  postsynaptic spikes
    #------------------------------------------------------------------------    
    fig = plt.figure(1001, figsize=(100, 40))
    
    ax0 = plt.subplot(2, 1, 1)
    sp_in = plotRaster(spikeTimes)
    plt.title('presynaptic spikes', fontsize = 100)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.ylabel("neuron index", fontsize = 80)
    
    ax1 = plt.subplot(2, 1, 2)
    sp_out = cg1Spikes[0].plot()
    plt.title('postsynaptic spikes', fontsize = 100)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.xlabel("", fontsize = 60)
    plt.ylabel("neuron index", fontsize = 80)

    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/overlap/PSTH_overlap" + str(numOverlapedPixels) + ".png"
        fig.savefig(fileName)
        print("No display available, saved to file " + fileName + ".")
    print("Saving data...")    
    data = [target_neurons, weightMatrix]    
    with open('data/data_Overlap.pkl', 'wb') as f:
        pickle.dump(data , f)
