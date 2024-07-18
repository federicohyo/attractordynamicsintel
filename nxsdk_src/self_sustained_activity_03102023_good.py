# Import modules

import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
from nxsdk.utils.plotutils import plotRaster

# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Set up the network architecture
def setupNetwork(net, numECores, numICores, numSpikeGen, tau, E1pc, Enoise, Inoise):

    # Create an excitatory compartment group consisting of numECores neuron compartments
    # cg1... excitatory population
    compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3 #3
            )
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
   

    # Create an inhibitory compartment group consisting of numICores neuron compartments
    # cg2... inhibitory population
    compartmentPrototype2 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3 
            )
        
    cg2 = net.createCompartmentGroup(size=numICores, prototype=compartmentPrototype2)

    # Set up the connection prototypes by specifying the weights (synaptic efficacies)
    connProto1 = nx.ConnectionPrototype(weight=35, delay=5) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=5)
    connProtoIn = nx.ConnectionPrototype(weight=30, delay=5) 
    connProtoRecurrent = nx.ConnectionPrototype(weight=25, delay=5) 
    connProtoNoise = nx.ConnectionPrototype(weight=10, delay=5) 

    # Connect input spike generators E1pc to the excitatory population (each spike generator connected to a neuron)
    synaptic_matrix0 = np.eye(numECores)
    E1pc.connect(cg1, prototype=connProtoIn, connectionMask=synaptic_matrix0) 
    
    # Connect E1pc and the inhibitory population
    synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
    E1pc.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix1) 
    
    # Connect noise generators to neurons
    Enoise.connect(cg1, prototype=connProtoNoise, connectionMask=np.eye(numECores)) 
    Inoise.connect(cg2, prototype=connProtoNoise, connectionMask=np.eye(numICores)) 
    
    # Create the recurrent connections for the excitatory population
    synaptic_connectivity2 = 0.22
    synaptic_matrix2 = np.bitwise_and(np.random.rand(numECores, numECores) < synaptic_connectivity2, \
                                      np.invert(np.eye(numECores, dtype=bool))*1)
    cg1.connect(cg1, prototype=connProtoRecurrent, connectionMask=synaptic_matrix2) 
    
    # Connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.29
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3) 
    
    # Recurrent connections of the inhibitory population
    synaptic_connectivity4 = 0.50
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    # Connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.21
    synaptic_matrix5 = np.random.rand(numECores, numICores) < synaptic_connectivity5
    cg2.connect(cg1, prototype=connProto2, connectionMask=synaptic_matrix5)
    
    # Configure probes
    SpikesProbe  = cg1.probe(nx.ProbeParameter.SPIKE, probeConditions = None)
    VoltageProbe = cg1[0].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None)

    return SpikesProbe, VoltageProbe

# Create numECores=numSpikeGen spike generators 
def generateSpikes(net, runTime, numSpikeGen, spikingProbability):
    lengthProb = len(spikingProbability)
    spikeTimeDiffFreq = []
    
    for probStep in range(lengthProb):
        x = genPoissonSpikeTimes(runTime//lengthProb, spikingProbability[probStep], numSpikeGen)
        spikeTimeDiffFreq.append(x.copy()) 
        if probStep == 0:
            spikeTimes = x
        else:
            for gen in range(len(x)):
                for i in range(len(x[gen])):
                    if x[gen] != []:
                        x[gen][i] += (probStep)*runTime//lengthProb
                if probStep > 0:
                    spikeTimes[gen] = spikeTimes[gen] + x[gen]
    
    spikeGen = createSpikeGenerators(net, numSpikeGen, spikeTimes)
   
    return spikeGen, spikeTimes, spikeTimeDiffFreq

# Create numSpikeGen spike generators
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   #generate spikes in numSpikeGen ports
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) #addSpikes(portID, spikeTimes)
    return spikeGen

# Produce spike times of numSpikeGen spike generators
def genPoissonSpikeTimes(runTime, spikingProbability, numSpikeGen):
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

# Calculate the mean spiking frequency of the input generators
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

# Calculate the mean post-synaptic spiking frequency of the excitatory population
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores):    
    numECorePerPop = numCores
    numPop = numCores//numECorePerPop
    window_size = 25
    window = np.ones(window_size) # Window size of 25 time steps for binning.
    buffer_in_avg = [[] for ii in range(numPop)]
 
    for pop in range(numPop):
        buffer = np.zeros(())
        for neuron in range(numECorePerPop):
            binned = (100//window_size)*np.asarray([np.convolve(postSynapticSpikes.data[pop*numECorePerPop+neuron], \
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
    
    #Number of excitatory neurons
    numECores = 128 
    #Number of inhibitory neurons
    numICores = 64
    #Number of input spike generators 
    numSpikeGen = numECores

    #Spiking probabilities of the input spikes at three different time intervals
    spikingProbability = [0.2, 0.6, 0.2] 
    #Spiking probabailities of the noise sources
    EnoiseSpikingProbability = 0.15
    InoiseSpikingProbability = 0.15
    #Voltage decay time constant
    tau = 16 
    
    for trial in range(1):
        # Initialize a network
        net = nx.NxNet()

        ### Configure the network

        # Create spike generators
        (Epc1, spikeTimes, spikeTimeDiffFreq) = generateSpikes(net, runTime, numSpikeGen, spikingProbability)
        spikeENoiseTimes = genPoissonSpikeTimes(runTime, EnoiseSpikingProbability, numECores) 
        spikeINoiseTimes = genPoissonSpikeTimes(runTime, InoiseSpikingProbability, numICores) 
        Enoise = createSpikeGenerators(net, numECores, spikeENoiseTimes) 
        Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes) 
        (cg1Spikes, cg1_0Voltage) = setupNetwork(net, numECores, numICores, numSpikeGen, tau, Epc1, Enoise, Inoise)

        # Execution   
        net.run(runTime, executionOptions={'partition':'nahuku32_2h'})
        net.disconnect()

        # Data Processing

        #Calculates the mean output frequency over time
        meanFreqOutArray = calculateMeanFreqOut(runTime, cg1Spikes[0], numECores) 

        # Plot
        fig = plt.figure(1001+trial, figsize=(60,80))

        ax0 = plt.subplot(4, 1, 1)
        sp_in = plotRaster(spikeTimes)
        #plt.title('presynaptic spikes', fontsize = 100)
        #plt.xticks(fontsize=60)
        #plt.yticks(fontsize=60)
        #plt.ylabel("neuron index", fontsize = 80)

        ax1 = plt.subplot(4, 1, 2)
        sp_in = cg1Spikes[0].plot()
        #plt.title('postsynaptic spikes', fontsize = 100)
        #plt.xticks(fontsize=60)
        #plt.yticks(fontsize=60)
        #plt.xlabel("", fontsize = 20)
        #plt.ylabel("neuron index", fontsize = 80)

        ax2 = plt.subplot(4, 1, 3)
        vol = cg1_0Voltage[0].plot()
        #plt.title('membrane voltage', fontsize = 100)
        #plt.xticks(fontsize=60)
        #plt.yticks(fontsize=60)
        #plt.xlabel("", fontsize = 20)
        #plt.ylabel("voltage", fontsize = 80)

        ax3 = plt.subplot(4, 1, 4)

        plt.plot(meanFreqOutArray[0])
        #plt.xlabel("time", fontsize = 100)
        #plt.ylabel("frequency [/100 time steps]", fontsize = 80)
        #plt.title('spiking activity of excitatory population over time', fontsize = 100)
        #plt.xticks(fontsize=60)
        #plt.yticks(fontsize=60)

        ax0.set_xlim(0, runTime)
        ax1.set_xlim(ax0.get_xlim())
        ax2.set_xlim(ax0.get_xlim())
        ax3.set_xlim(ax0.get_xlim())

        # Save the figure
        if haveDisplay:
            plt.show()
        else:
            fileName = "figures/self_sustained_activity/self_sustained_activity.png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)
    
        
if __name__ == "__main__":        
    main()
