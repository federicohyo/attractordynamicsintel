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
import pandas as pd
import math
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


def setupNetwork(net, numECores, tau, spikeGen, rD):
    
    # -------------------------------------------------------------------------
    # Create numECores compartments and make a compartment group
    # Make an excitatory population
    # -------------------------------------------------------------------------
    #cg1... excitatory population
    
    compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=300,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = rD
            )
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
    
    connProto1 = nx.ConnectionPrototype(weight=30, delay=0)  #connection prototype, no learning
    
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
    spikeGen.connect(cg1, prototype=connProto1, connectionMask=synaptic_matrix0) 
   
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
    sumVar = 0
    diffFreq = []
    variance = 0
    
    for generator in range(numSpikeGen):
        spikeTimePerGen = spikeTimes[generator]
        if len(spikeTimePerGen) > 1:
            for spike in range(len(spikeTimePerGen)-1):
                diffSpikeTime = (spikeTimePerGen[spike+1] - spikeTimePerGen[spike])*0.01
                diffCount += 1 
                sumPeriod += diffSpikeTime
                diffFreq.append(1/diffSpikeTime)
                
    if diffCount == 0:
        aveFreq = 0
    else:
        avePeriod = sumPeriod/diffCount
        aveFreq = 1/avePeriod
        
    for freq in range(len(diffFreq)):
        sumVar += (diffFreq[freq] - aveFreq)**2
    if len(diffFreq) > 0:
        variance = sumVar/len(diffFreq)  
    
    return aveFreq, variance

# calculate the mean frequency of output spikes from a probe of size = numCores
def calculateMeanFreqOut(postSynapticSpikes, numCores):
    sumPeriod = 0
    diffCount = 0
    spikeTimes = []
    sumVar = 0
    diffFreq = []
    variance = 0
    for neuron in range(numCores):
        spikeTimes.append(np.where(postSynapticSpikes.data[neuron])[0].tolist())
        if len(spikeTimes[neuron]) > 1:
            for spike in range(len(spikeTimes[neuron])-1):
                diffSpikeTime = (spikeTimes[neuron][spike+1] - spikeTimes[neuron][spike])*0.01
                sumPeriod += diffSpikeTime
                diffFreq.append(1/diffSpikeTime)
                diffCount += 1
                
    if diffCount == 0:
        aveFreq = 0
    else:
        aveFreq = diffCount/sumPeriod
        
    for freq in range(len(diffFreq)):
        sumVar += (diffFreq[freq] - aveFreq)**2
    if len(diffFreq) > 0:
        variance = sumVar/len(diffFreq)  
  
    return aveFreq, variance
            
        
        
    
        
    
# -----------------------------------------------------------------------------
# Run the tutorial
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    
    net = nx.NxNet()   # Create a network
    
    runTime = 3000 #simulation time
    
    
    numCores = 128   #Number of neurons
    numSpikeGen = 512  #Number of inputs per neuron 
    
    tau = 20
    
    spikingProbability = [0, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    numProb = len(spikingProbability)
    #refractory period of the input spikes
    tEpoch = 1
    #refractory period of the output spikes
    refractoryDelay = 3
    
    # -------------------------------------------------------------------------
    # Configure network
    # -------------------------------------------------------------------------

    spikeGen = []
    spikeTimes = []
    net = nx.NxNet()
    cg1Spikes = [[] for _ in range(numProb)]

    for probStep in range(numProb):
        #---------------------------------------------------------------------
        # Configure network
        #--------------------------------------------------------------------
        (sG, sT) = generateSpikes(net, runTime, numSpikeGen, tEpoch, spikingProbability[probStep])
        spikeGen.append(sG)
        spikeTimes.append(sT)
        cg1Spikes[probStep] = setupNetwork(net, numCores, tau, spikeGen[probStep], refractoryDelay)

    #------------------------------------------------------------------------
    # Run    
    #------------------------------------------------------------------------
    net.run(runTime, executionOptions={'partition':'nahuku32'})
    net.disconnect()
    #------------------------------------------------------------------------
    # Data Processing
    #------------------------------------------------------------------------
    meanFreqInArray = np.zeros(numProb)
    meanFreqOutArray = np.zeros(numProb)
    variance_in = np.zeros(numProb)
    variance_out = np.zeros(numProb)
    for probStep in range(numProb):
        #Calculates the mean input frequency for different spiking probabilities set for poisson spike generators
        (meanFreqInArray[probStep], variance_in[probStep]) = calculateMeanFreqIn(spikeTimes[probStep], numSpikeGen)
        #Calculates the mean output frequency for different input frequencies
        #index 0 to extract object "ProbeSet"
        (meanFreqOutArray[probStep], variance_out[probStep]) = calculateMeanFreqOut(cg1Spikes[probStep][0], numCores) 
        print("Average Input Frequency is ", float(meanFreqInArray[probStep]), " per 100 timestep.")
        print("Average Output Frequency is ", float(meanFreqOutArray[probStep]), " per 100 timestep.")
        print("----------------------------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    #Neuronal TF 
    fig = plt.figure(1001, figsize=(25,25))
    plt.xlabel('input spiking frequency [/100 timesteps]', fontsize = 30)
    plt.ylabel('output spiking frequency [/100 timesteps]', fontsize = 30)
    plt.title('neuron transfer function', fontsize = 30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(meanFreqInArray, meanFreqOutArray)
    plt.errorbar(meanFreqInArray, meanFreqOutArray, yerr=variance_out**0.5, fmt='o', capsize=5)    
       
    '''ax0 = plt.subplot(2, 1, 1)
    sp_in = plotRaster(spikeTimes[0])
    plt.title('presynaptic spikes')

    ax1 = plt.subplot(2, 1, 2)
    sp_out = cg1Spikes[5].plot()
    plt.title('post-synaptic spikes')
    
    ax0.set_xlim(ax1.get_xlim())
    ax1.set_xlim(ax1.get_xlim())
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())'''
    
    #-----------------------------------------------------------------------------------------------
    # Theory
    #-----------------------------------------------------------------------------------------------
    CurrentLeakage = 1/tau
    mean_ext = meanFreqInArray*0.01

    mean =  meanFreqInArray - CurrentLeakage
    variance = variance_in*0.01
    vThMant=300      
    vTHexp = 0
    theta = vThMant
    H = 0
    vout = np.zeros(numProb)
    for ii in range(numProb):
        if mean[ii] == 0 or variance[ii] == 0:
            vout[ii] = 0
        else:
            vout[ii] = (refractoryDelay+variance[ii]/(2*mean[ii]**2)*(math.exp(-2*mean[ii]*(theta - H)/variance[ii])-1+2*mean[ii]*(theta-H)/variance[ii]))**-1
        print(vout[ii] )
    #plt.figure(1001)
    #plt.plot(meanFreqInArray, vout)
    
    
    headers = {'FreqIn': 'mean input freq', 'FreqOut': 'mean output freq', 'TheoryFreqOut': 'theoretical mean output freq'}
    data = {'FreqIn': meanFreqInArray, 'FreqOut': meanFreqOutArray, 'TheoryFreqOut': vout}
    df = pd.DataFrame(data)
    
    df.to_excel('data/NeuronTF.xlsx', index=False, header = headers)
    
    if haveDisplay:
        plt.show()
    else:
        fileName = "figures/NeuronTF.png"
        print("No display available, saving to file " + fileName + ".")
        fig.savefig(fileName)
