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
import math

from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition

haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

os.environ['PARTITION'] = "nahuku32"
os.environ['BOARD'] = "ncl-ext-ghrd-01"

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
            logicalCoreId = coreID 
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
                logicalCoreId = coreiID 
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
                                                weightLimitMant=4,
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
    synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
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
    board = nx.N2Compiler().compile(net)
    return board

# Create numECores=numSpikeGen spike generators 
def generateSpikes(net, runTime, inputFraction, numSpikeGen, numESubPop, tEpoch, spikingProbability, repetition, reset):
    spikeTimes = genPoissonSpikeTimes(runTime, 
                                      inputFraction,
                                      spikingProbability, 
                                      tEpoch, 
                                      numSpikeGen, 
                                      numESubPop, 
                                      repetition, 
                                      reset)
    spikeGen = createSpikeGenerators(net, numSpikeGen, spikeTimes)  
    return spikeGen, spikeTimes

# Create numSpikeGen spike generators with spike timing, spikeTimes
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) 
    return spikeGen

# Produce spike timing of numSpikeGen spike generators
def genInputSpikeTimes(runTime, spikingProbability, tEpoch, numSpikeGen, numESubPop, repetition):
    """Generates an approximate Poisson spike train with a refractory period of tEpoch after each spike in order to avoid multiple spikes within the same learning epoch."""
    runTimeRep = runTime//repetition
    spikeTimes = []
    initialIdleTime = 300
    spikes = np.zeros((numSpikeGen, runTime)) 
    numNeuronsPerSubpop = round(numSpikeGen/numESubPop)
    # Add input and noise to the matrix, spikes
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
    
    maxBufferSize = 1024*2
    binSize = 16
    runTime = maxBufferSize*binSize
    
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
    ImpParam = [20, 20, 20, 20, 20]
    x1Tau=4
    y1Tau=4
    
    # Sample cycle of weights
    dt = runTime//4
    
    #--------------------------------------------------------------------
    # Generate spikes or load spikes from the last execution
    #--------------------------------------------------------------------   
    spikeTimes = genInputSpikeTimes(runTime, 
                                      inputSpikingProbability, 
                                      tEpoch, 
                                      numSpikeGen, 
                                      numESubPop, 
                                      repetition)
    spikeResetTimes = genResetSpikeTimes(runTime, 32, numESubPop, repetition)
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
        board = setupNetwork(net, 
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
        power_neuron_logic_static.append(powerStats.power['core']['logic_static'])
        power_neuron_SRAM_static.append(powerStats.power['core']['SRAM_static'])
        
        power_total.append(powerStats.power['total'])
        power_total_static.append(powerStats.power['static'])
        power_total_dynamic.append(powerStats.power['dynamic'])
        print("timePerTimestep:", timePerTimestep[trial])


    data = {}
    data['timePerTimestep'] = timePerTimestep
    data['power_total'] =  power_total
    data['power_total_static'] = power_total_static
    data['power_total_dynamic'] =  power_total_dynamic
    data['power_neuron_static'] = power_neuron_static
    data['power_neuron_dynamic'] =  power_neuron_dynamic
    data['power_neuron_logic_static'] = power_neuron_logic_static
    data['power_neuron_SRAM_static'] =  power_neuron_SRAM_static
    
    '''if haveDisplay:
        plt.show()
    else:
        fileName = "figures/energy/energy_learning.png"
        fig.savefig(fileName)
        print("No display available, saved to file " + fileName + ".")'''


    df = pd.DataFrame(data) 
    df.to_excel('data/energy/energy_learning.xlsx', index=False)
    
if __name__ == "__main__":        
    main()