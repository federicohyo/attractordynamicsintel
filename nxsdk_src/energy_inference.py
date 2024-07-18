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
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition

os.environ['PARTITION'] = "nahuku32"
os.environ['BOARD'] = "ncl-ext-ghrd-01"

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

def generateTarget(numECores, numNeuronsSubPop, errorPortion):
    if numNeuronsSubPop > numECores:
        print("Error: Subpopulation is too large.")
        return 0
    target_neurons = []
    numSubPop = numECores//numNeuronsSubPop
    stimulusPortion = 1 - errorPortion
    for pop in range(numSubPop):
        neurons = np.arange(pop*numNeuronsSubPop, (pop+stimulusPortion)*numNeuronsSubPop)
        flipped_neuron_array = np.array([])
        for flipped_bits in range(round(numNeuronsSubPop*errorPortion)):
            if pop == 0:
                flipped_neuron = random.randint(numNeuronsSubPop, numECores-1)
            elif pop == 3:
                flipped_neuron = random.randint(0, 3*numNeuronsSubPop-1)
            else:
                if random.choice([True, False]):
                    flipped_neuron = random.randint(0, pop*numNeuronsSubPop-1)
                else:
                    flipped_neuron = random.randint((pop+1)*numNeuronsSubPop, numECores-1)
            if np.any(flipped_neuron_array) == flipped_neuron:
                flipped_bits = flipped_bits - 1
            else:
                flipped_neuron_array = np.append(flipped_neuron_array, flipped_neuron)
        neurons = np.append(neurons, flipped_neuron_array).astype(int)
        target_neurons.append(neurons) 
        print(neurons)
    return target_neurons


def setupNetwork(net, 
                  numECores,
                  numEPerCore,
                  numICores,  
                  numIPerCore,
                  numSpikeGen, 
                  tau,           
                  E1pc, 
                  Enoise,
                  Inoise,
                  Ireset,
                  weightMatrix_trained,            
                  tEpoch):
    # -------------------------------------------------------------------------
    # Create numECores compartments and make a compartment group
    # Make an excitatory population
    # -------------------------------------------------------------------------
    #cg1... excitatory population
    cg1 = net.createCompartmentGroup(size=0) 
    Ecg = []

    for coreID in range(numECores//numEPerCore):
        compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3,
            logicalCoreId = coreID 
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
                logicalCoreId = coreiID 
                )
        cg2.addCompartments(net.createCompartmentGroup(size=numIPerCore, prototype=compartmentPrototype2))

    
    #Other connection prototypes
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=0)
    connProtoIn = nx.ConnectionPrototype(weight=30, delay=0)
    connProtoNoise = nx.ConnectionPrototype(weight=30, delay=0) 
    connProtoReset = nx.ConnectionPrototype(weight=30, delay=0) 
    
    #connect E1pc to each excitatory neuron
    synaptic_matrix0 = np.eye(numECores)
    E1pc.connect(cg1, prototype=connProtoIn, connectionMask=synaptic_matrix0) 
    
    #connect noise to neurons
    Enoise.connect(cg1, prototype=connProtoNoise, connectionMask=np.eye(numECores)) 
    Inoise.connect(cg2, prototype=connProtoNoise, connectionMask=np.eye(numICores)) 
    
    #connect Ireset and the inhibitory population (reset)
    Ireset.connect(cg2, prototype=connProtoReset)
    
    #connect E1pc and the inhibitory population
    synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
    E1pc.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix1) 
   
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
    for row in range(numECores):
        for col in range(numECores):
            if weightMatrix_trained[row, col] > 0:
                connProtoRecurrent = nx.ConnectionPrototype(weight=weightMatrix_trained[row, col], delay=0) 
                cg1[col].connect(cg1[row], prototype=connProtoRecurrent)

    '''VoltageProbe=[]
    for pop in range(numECores//128):
        VoltageProbe.append(cg1[pop*128].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None))
    SpikesProbe = cg1.probe(nx.ProbeParameter.SPIKE, probeConditions = None)'''
    VoltageProbe = None
    SpikesProbe = None
    board = nx.N2Compiler().compile(net)
    return board, VoltageProbe, SpikesProbe
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
                          spikingProbability, 
                          tEpoch):
    runTime = np.size(spikeArray, axis = 1)
    numSpikeGen = np.size(spikeArray, axis = 0)
    startTime = initialIdleTime+stimulus_number*(stimulation_interval+reset_interval)
    binSize = 100
    nbin = stimulation_interval//binSize
    
    for neuron in target_neurons:
        for ii in range(nbin):
            spikeArray[neuron, startTime+ii*binSize:startTime+(ii+1)*binSize] = np.random.rand(binSize) < spikingProbability[1]
            
    spikeTimes = []
    for neuron in range(numSpikeGen):
        spikeTimes.append(np.where(spikeArray[neuron, :])[0].tolist()) 
    return spikeTimes

def genResetSpikeTimes(runTime, numSpikeGen, stimulation_interval, resting_interval):
    runTimeRep = runTime
    total_interval = stimulation_interval + resting_interval
    spikeTimes = []
    spikes = np.zeros((numSpikeGen, runTime)) 
    count = 1
    for gen in range(numSpikeGen):
        count = 1
        for time in range(runTime):
            if time >= int(total_interval*count - resting_interval*0.25)+initialIdleTime \
                          and time < total_interval*count+initialIdleTime:
                spikes[gen][time] = 1
            if time == total_interval*count+initialIdleTime:
                count += 1
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

# Calculate the mean frequency of output spikes from a probe of size = numCores for each sample period
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores, numNeuronsSubPop):    
    numPop = numCores//numNeuronsSubPop
    window_size = 25
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

def attractor_test(meanFreqOutArray, stimulation_interval, resting_interval):
    threshold = 10
    total_interval = stimulation_interval + resting_interval
    test_result = []
    for ii in range(len(meanFreqOutArray)):
        test_result_buffer = 1 
        startTime = initialIdleTime + stimulation_interval + 50 + total_interval*ii 
        endTime = int(initialIdleTime + stimulation_interval - resting_interval*0.25 + total_interval*ii + resting_interval)
        for time in range(startTime, endTime):
            if meanFreqOutArray[ii][time] < threshold:
                test_result_buffer = 0 
                break
        test_result.append(test_result_buffer)
    return test_result
# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
initialIdleTime = 300

def main():    
    with open('data/data_weight_multi0.pkl', 'rb') as file:
        weightMatrix = pickle.load(file)
    weightMatrix_trained = np.nan_to_num(weightMatrix[len(weightMatrix)-1], nan=0)
    # Time duration of the execution
    maxBufferSize = 1024
    binSize = 4
    runTime = maxBufferSize*binSize
    print("The total number of timesteps is ", runTime, ".")

    numECores = 512
    numNeuronsSubPop = 128 
      
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = [0.1, 0.8]
    
    # Number of excitatory neurons per core
    numEPerCore = 128
    numESubPop = numECores//numNeuronsSubPop
    # Number of inhibitory neurons
    numICores = numECores//2
    numIPerCore = 64
    # Number of generators
    numSpikeGen = numECores
    # Time constant
    tau=16
    # Refractory period of the input spikes
    tEpoch = 1
    stimulation_interval = (runTime-initialIdleTime)//(2*numESubPop)
    resting_interval = (runTime-initialIdleTime)//(2*numESubPop)
    errorPortion = [0, 0, 0, 0, 0]
    
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.5
    spikeENoiseTimes = genNoiseSpikeTimes(runTime, EnoiseSpikingProbability, numECores)
    spikeINoiseTimes = genNoiseSpikeTimes(runTime, InoiseSpikingProbability, numICores)
    timePerTimestep = []
    power_neuron_dynamic = []
    power_neuron_static = []
    power_total = []
    power_total_static = []
    power_total_dynamic = []
    power_neuron_logic_static = []
    power_neuron_SRAM_static = []
    for ii in range(len(errorPortion)):
        spikeResetTimes = genResetSpikeTimes(runTime, 32, stimulation_interval, resting_interval)
        accuracy = np.zeros(numESubPop)
        test_sum = 0
        
        target_neurons = generateTarget(numECores, 
                                    numNeuronsSubPop, 
                                    errorPortion[ii])  
        # Initialize spikeArray with noiseSpikingProbability
        spikeArray = np.random.rand(numSpikeGen, runTime) < inputSpikingProbability[0] 
        for stimulus_number in range(len(target_neurons)):
            spikeTimes_buffer = genStimulusSpikeTimes(spikeArray,
                                                    target_neurons[stimulus_number], 
                                                    stimulus_number,
                                                    stimulation_interval,
                                                    resting_interval,
                                                    inputSpikingProbability, 
                                                    tEpoch)
            if stimulus_number == 0:
                spikeTimes = spikeTimes_buffer
                spikeArray = np.zeros((numSpikeGen, runTime))
            else:
                for gen in range(numSpikeGen):
                    spikeTimes[gen] = spikeTimes[gen]+spikeTimes_buffer[gen]
        net = nx.NxNet()        
        Epc1 = createSpikeGenerators(net, numSpikeGen, spikeTimes) 
        Enoise = createSpikeGenerators(net, numSpikeGen, spikeENoiseTimes) 
        Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes) 
        Ireset = createSpikeGenerators(net, 32, spikeResetTimes)
        #--------------------------------------------------------------------
        # Configure network
        #--------------------------------------------------------------------   
        print("ErrorPercentage:", 100*errorPortion[ii])
        print("Spikes were generated. Configuring network...")    
        (board, cg1_0Voltage, cg1Spikes) = setupNetwork(net, 
                              numECores,
                              numEPerCore,
                              numICores,  
                              numIPerCore,
                              numSpikeGen, 
                              tau,           
                              Epc1, 
                              Enoise,
                              Inoise,
                              Ireset,
                              weightMatrix_trained,            
                              tEpoch)
        print("Board compiled.")

        # Create energy probe
        pc = PerformanceProbeCondition(tStart=1, tEnd=runTime, bufferSize=maxBufferSize, binSize=binSize)
        eProbe = board.probe(ProbeParameter.ENERGY, pc)
        
        # Run network
        board.run(runTime)
        board.disconnect()        
        powerStats = board.energyTimeMonitor.powerProfileStats
        print("Execution finished.")
        
        timePerTimestep.append(powerStats.timePerTimestep)
        power_neuron_dynamic.append(powerStats.power['core']['dynamic'])
        power_neuron_static.append(powerStats.power['core']['static'])
        power_total.append(powerStats.power['total'])
        power_total_static.append(powerStats.power['static'])
        power_total_dynamic.append(powerStats.power['dynamic'])
        power_neuron_logic_static.append(powerStats.power['core']['logic_static'])
        power_neuron_SRAM_static.append(powerStats.power['core']['SRAM_static'])
        print("timePerTimestep:", timePerTimestep[ii])
        


    data = {}
    data['timePerTimestep'] = timePerTimestep
    data['power_total'] =  power_total
    data['power_total_static'] = power_total_static
    data['power_total_dynamic'] =  power_total_dynamic
    data['power_neuron_static'] = power_neuron_static
    data['power_neuron_dynamic'] =  power_neuron_dynamic
    data['power_neuron_logic_static'] = power_neuron_logic_static
    data['power_neuron_SRAM_static'] =  power_neuron_SRAM_static
            
    df = pd.DataFrame(data) 
    df.to_excel('data/energy/energy_inference.xlsx', index=False)
        
if __name__ == "__main__":          
    main()