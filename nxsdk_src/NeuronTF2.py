"""
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2018-2022 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express 
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy, 
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are 
expressly stated in the License.
"""

# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------

# For plotting without GUI
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
from nxsdk.utils.plotutils import plotRaster

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


def setupNetwork(net, numECores, numICores, numSpikeGen, tau, E1pc, rD, coreID):
    # -------------------------------------------------------------------------
    # Create numECores compartments and make a compartment group
    # Make an excitatory population
    # -------------------------------------------------------------------------
    #cg1... excitatory population
    
    compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=300,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = rD,
            logicalCoreId = coreID
            )
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
    
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0)  #connection prototype, no learning
    
    # -------------------------------------------------------------------------
    # Make an inhibitory population 
    # -------------------------------------------------------------------------
    #cg2... inhibitory population
    compartmentPrototype2 = nx.CompartmentPrototype(
            vThMant=300,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3,
            logicalCoreId = coreID
            )
        
    cg2 = net.createCompartmentGroup(size=numICores, prototype=compartmentPrototype2)

    connProto1 = nx.ConnectionPrototype(weight=30, delay=0, signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=0, signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY)
    #connect E1pc and the excitatory population (each spike generator connected to a neuron)
    #uniform connection with connectivity 10% (each external spike generator connects to numECores//50 neurons which are chosen randomly)
    synaptic_matrix0 = np.zeros((numECores, numSpikeGen)) 
    count = 0
    random_int = []
    for row in range(numECores):
        while count != numECores//10:
            val = np.random.randint(0, numSpikeGen)
            if val not in random_int:
                random_int.append(val)
                synaptic_matrix0[row][random_int[count]] = 1
                count = count + 1
        count = 0
        random_int = []
    E1pc.connect(cg1, prototype=connProto1, connectionMask=synaptic_matrix0) 
    
    #connect E1pc and the inhibitory population
    synaptic_connectivity1 = 0.1
    synaptic_matrix1 = np.random.rand(numICores, numSpikeGen) < synaptic_connectivity1
    E1pc.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix1) 
    
    #connect recurrent stimulus and the excitatory population
    synaptic_connectivity2 = 0.25
    synaptic_matrix2 = np.random.rand(numECores, numECores) < synaptic_connectivity2
    cg1.connect(cg1, prototype=connProto1, connectionMask=synaptic_matrix2) 
    
    #connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.2
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3) 
    
    #recurrent connection of the inhibitory population
    synaptic_connectivity4 = 0.52
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    #connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.25
    synaptic_matrix5 = np.random.rand(numECores, numICores) < synaptic_connectivity5
    cg2.connect(cg1, prototype=connProto2, connectionMask=synaptic_matrix5)
    
   
    # -------------------------------------------------------------------------
    # Configure probes
    # -------------------------------------------------------------------------
    probeConditions = None
    probeParameters = [nx.ProbeParameter.SPIKE]
    SpikesProbe  = cg1.probe(probeParameters, probeConditions = None)
    

    return SpikesProbe
    
# -------------------------------------------------------------------------
    #Create numECores=numSpikeGen spike generators 
# -------------------------------------------------------------------------
def generateSpikes(net, runTime, numSpikeGen, tEpoch, spikingProbability):
    spikeTimes = []
    spikeGen = []
   
    #spikeTimes size = [numProb, numSpikeGen]
    spikeTimes = genPoissonSpikeTimes(runTime, spikingProbability, tEpoch, numSpikeGen)
    spikeGen = createSpikeGenerators(net, numSpikeGen, spikeTimes)
   
    return spikeGen, spikeTimes
    
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
     #Create numSpikeGen spike generators
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   #generate spikes in numSpikeGen ports
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) #addSpikes(portID, spikeTimes)
    return spikeGen

def genPoissonSpikeTimes(runTime, spikingProbability, tEpoch, numSpikeGen):
    """Generates an approximate Poisson spike train with a refractory period of tEpoch after each spike in order to avoid
    multiple spikes within the same learning epoch."""
    
    spikes = np.zeros((runTime, numSpikeGen))
    spikeTimes = []
    refCtr = 0
    
    for port in range(numSpikeGen):
        for i in range(runTime):
            spikes[i][port] = (np.random.rand(1) < spikingProbability) and refCtr <= 0
            
            if spikes[i][port]:
                refCtr = tEpoch + 1
            refCtr -= 1
        
        spikeTimes.append(np.where(spikes[:, port])[0].tolist()) #[0] to extract row indices, [1] to extract column indices
    return spikeTimes

# calculate the mean frequency of input spikes from a list of size = (numCores, numSpikeGen)
def calculateMeanFreqIn(spikeTimes, numSpikeGen):
    sumPeriod = 0 
    diffCount = 0
    diffSpikeTime = 0
    
    for generator in range(numSpikeGen):
        spikeTimePerGen = spikeTimes[generator]
        if len(spikeTimePerGen) > 1:
            for spike in range(len(spikeTimePerGen)-1):
                diffSpikeTime = (spikeTimePerGen[spike+1] - spikeTimePerGen[spike])*0.01
                diffCount += 1 
                sumPeriod += diffSpikeTime
                
    if diffCount == 0:
        aveFreq = 0
    else:
        avePeriod = sumPeriod/diffCount
        aveFreq = 1/avePeriod

    return aveFreq

# calculate the mean frequency of output spikes from a probe of size = numCores
def calculateMeanFreqOut(postSynapticSpikes, numCores):
    sumPeriod = 0
    diffCount = 0
    spikeTimes = []
    sumVariance = 0
    varFreq = 0
    freqSamples = []
    
    for neuron in range(numCores):
        spikeTimes.append(np.where(postSynapticSpikes.data[neuron])[0].tolist())
        if len(spikeTimes[neuron]) > 1:
            for spike in range(len(spikeTimes[neuron])-1):
                diffSpikeTime = (spikeTimes[neuron][spike+1] - spikeTimes[neuron][spike])*0.01
                freqSamples.append(1/diffSpikeTime)
                sumPeriod += diffSpikeTime
                diffCount += 1

                
    if diffCount == 0:
        aveFreq = 0
    else:
        aveFreq = diffCount/sumPeriod
    
    if len(freqSamples) > 0:
        for freq in range(len(freqSamples)):    
            sumVariance += (freqSamples[freq] - aveFreq)**2
        varFreq = sumVariance/len(freqSamples)
            
    return aveFreq, varFreq
            
        
        
    
        
    
# -----------------------------------------------------------------------------
# Run the tutorial
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    
    net = nx.NxNet()   # Create a network
    
    runTime = 3000   #simulation time
    
    
    numECores = 128   #Number of neurons
    numICores = 64
    numSpikeGen = 512 #Number of inputs per neuron 
    
    tau = 20
    
    spikingProbability = [0,0.04, 0.07, 0.1, 0.15, 0.17, 0.19, 0.2]
    numProb = len(spikingProbability)
    #refractory period of the input spikes
    tEpoch = 0
    #refractory period of the output spikes
    refractoryDelay = 1
    
    # -------------------------------------------------------------------------
    # Configure network
    # -------------------------------------------------------------------------

    E1pc = []
    spikeTimes = []
    net = nx.NxNet()
    cg1Spikes = [[] for _ in range(numProb)]

    for probStep in range(numProb):
        #---------------------------------------------------------------------
        # Configure network
        #--------------------------------------------------------------------
        (sG, sT) = generateSpikes(net, runTime, numSpikeGen, tEpoch, spikingProbability[probStep])
        E1pc.append(sG)
        spikeTimes.append(sT)
        cg1Spikes[probStep] = setupNetwork(net, numECores, numICores, numSpikeGen, tau, E1pc[probStep], refractoryDelay, probStep)
    #------------------------------------------------------------------------
    # Run    
    #------------------------------------------------------------------------
    net.run(runTime, executionOptions={'partition':'nahuku32'})
    net.disconnect()
    #------------------------------------------------------------------------
    # Data Processing
    #------------------------------------------------------------------------
    meanFreqInArray = np.zeros((numProb, 1))
    varFreq = np.zeros((numProb, 1))
    meanFreqOutArray = np.zeros((numProb, 1))
    stdFreq = np.zeros(numProb)
    for probStep in range(numProb):
        #Calculates the mean input frequency for different spiking probabilities set for poisson spike generators
        meanFreqInArray[probStep] = calculateMeanFreqIn(spikeTimes[probStep], numSpikeGen)
        #Calculates the mean output frequency for different input frequencies
        #index 0 to extract object "ProbeSet"
        (meanFreqOutArray[probStep], varFreq[probStep]) = calculateMeanFreqOut(cg1Spikes[probStep][0], numECores) 
        stdFreq[probStep] = varFreq[probStep]**0.5
        #print("Average Input Frequency is ", float(meanFreqInArray[probStep]), " per 100 timestep.")
        #print("Average Output Frequency is ", float(meanFreqOutArray[probStep]), " per 100 timestep.")
        #print("----------------------------------------------------------------------------")
    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    #Neuronal TF 
    fig = plt.figure(1001, figsize=(25,25))
    plt.xlabel('input spiking frequency [/100 timesteps]', fontsize = 30)
    plt.ylabel('output spiking frequency [/100 timesteps]', fontsize = 30)
    #plt.xlim(0, 100)
    #plt.ylim(0, 55)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(meanFreqInArray, meanFreqOutArray)
    #plt.scatter(meanFreqInArray, meanFreqOutArray)    
    plt.errorbar(meanFreqInArray, meanFreqOutArray, yerr=stdFreq, fmt='o', capsize=3)
   
    
    #-----------------------------------------------------------------------------------------------
    # Theory
    #-----------------------------------------------------------------------------------------------
    '''meanFreqInTheory = range(0, 100)
    CurrentLeakage = 1/tau
    mean_ext = meanFreqInArray
    Cee = 0.25
    Ne = numECores
    Jee = 30
    Jei = -30
    Ve = meanFreqInArray
    Vi = meanFreqInArray
    mean = Cee*Ne*Jee*Ve - Cei*Ni*Jei*Vi + mean_ext - CurrentLeakage
    variance = Cee*Ne*Jee**2*Ve + Cei*Ni*Jei**2*Vi + var_ext**2
    vThMant=300      
    vTHexp = 0
    theta = vThMant*2^(6+vTHexp)
    H = 0
    vout = (refractoryDelay+)'''
    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/NeuronalTF.png"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)
