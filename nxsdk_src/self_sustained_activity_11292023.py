# Import modules

import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
from nxsdk.utils.plotutils import plotRaster
import random
random.seed(42)
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
            refractoryDelay = 3,  #3
            vMinExp = 0, # -2^(vMinExp)+1
            logicalCoreId = 0
            )
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
   

    # Create an inhibitory compartment group consisting of numICores neuron compartments
    # cg2... inhibitory population
    compartmentPrototype2 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            refractoryDelay = 3,
            vMinExp = 0,
            logicalCoreId = 1
            )
        
    cg2 = net.createCompartmentGroup(size=numICores, prototype=compartmentPrototype2)

    # Set up the connection prototypes by specifying the weights (synaptic efficacies)
    connProto1 = nx.ConnectionPrototype(weight=35, delay=5) 
    connProto2 = nx.ConnectionPrototype(weight=-30, delay=5)
    connProtoIn = nx.ConnectionPrototype(weight=30, delay=5) 
    connProtoRecurrent = nx.ConnectionPrototype(weight=21, delay=5) 
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
    synaptic_connectivity2 = 0.257
    synaptic_matrix2 = np.bitwise_and(np.random.rand(numECores, numECores) < synaptic_connectivity2, \
                                      np.invert(np.eye(numECores, dtype=bool))*1)
    cg1.connect(cg1, prototype=connProtoRecurrent, connectionMask=synaptic_matrix2) 
    
    # Connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.30
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3) 
    
    # Recurrent connections of the inhibitory population
    synaptic_connectivity4 = 0.50
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    # Connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.20
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
def calculateMeanFreqOut(runTime, postSynapticSpikes, numCores, windowsize=25):    
    numECorePerPop = numCores
    numPop = numCores//numECorePerPop
    window_size = windowsize
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

def plot_psth(neural_data):
    """
    Plot a Peri-Stimulus Time Histogram (PSTH) for neural firing data.

    Parameters:
    neural_data (numpy.array): A 2D numpy array of shape (neurons, time_steps)
                               where each element is either 0 (no spike) or 1 (spike).
    """
    if neural_data.ndim != 2 or neural_data.shape[0] != 128:
        raise ValueError("Input must be a 2D numpy array with 128 neurons.")

    # Summing spikes across all neurons for each time step
    spike_counts = np.sum(neural_data, axis=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(range(neural_data.shape[1]), spike_counts, width=1.0)
    plt.xlabel('Time Steps')
    plt.ylabel('Spike Count')
    plt.title('Peri-Stimulus Time Histogram (PSTH)')
    plt.show()
# Example usage
# neural_data = np.random.randint(0, 2, (128, 1500))  # Example binary data (0s and 1s)
# plot_psth(neural_data) 

def plot_mean_population_activity(neural_data):
    """
    Plot the mean average population activity per time step for neural firing data.

    Parameters:
    neural_data (numpy.array): A 2D numpy array of shape (neurons, time_steps)
                               where each element is either 0 (no spike) or 1 (spike).
    """
    if neural_data.ndim != 2 or neural_data.shape[0] != 128:
        raise ValueError("Input must be a 2D numpy array with 128 neurons.")

    # Calculating mean average population activity per time step
    mean_activity = np.mean(neural_data, axis=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(mean_activity)
    plt.xlabel('Time Steps')
    plt.ylabel('Mean Population Activity')
    plt.title('Mean Population Activity Over Time')
    plt.grid(True)
    plt.show()
    
def moving_average(data, window_size):
    """
    Compute the moving average using a simple linear window.

    Parameters:
    data (numpy.array): 1D array of data to smooth.
    window_size (int): The size of the moving average window.

    Returns:
    numpy.array: Smoothed data.
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def plot_smoothed_average_spikes(neural_data, window_size):
    """
    Plot the smoothed average number of spikes per neuron per time step.

    Parameters:
    neural_data (numpy.array): A 2D numpy array of shape (neurons, time_steps)
                               where each element is either 0 (no spike) or 1 (spike).
    window_size (int): The size of the moving average window for smoothing.
    """
    if neural_data.ndim != 2 or neural_data.shape[0] != 128:
        raise ValueError("Input must be a 2D numpy array with 128 neurons.")

    # Calculating average number of spikes per neuron per time step
    average_spikes = np.mean(neural_data, axis=0)

    # Smoothing the average spikes
    smoothed_spikes = moving_average(average_spikes, window_size)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_spikes)
    plt.xlabel('Time Steps')
    plt.ylabel('Smoothed Average Spikes per Neuron')
    plt.title('Smoothed Average Spikes per Neuron per Time Step')
    plt.grid(True)
    plt.show()

    
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
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1
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
        plt.title('presynaptic spikes', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("neuron index", fontsize = 20)

        ax1 = plt.subplot(4, 1, 2)
        sp_in = cg1Spikes[0].plot()
        plt.title('postsynaptic spikes', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("", fontsize = 20)
        plt.ylabel("neuron index", fontsize = 20)

        ax2 = plt.subplot(4, 1, 3)
        vol = cg1_0Voltage[0].plot()
        plt.title('membrane voltage', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("", fontsize = 20)
        plt.ylabel("voltage", fontsize = 20)

        ax3 = plt.subplot(4, 1, 4)

        plt.plot(meanFreqOutArray[0])
        plt.xlabel("time", fontsize = 20)
        plt.ylabel("frequency [/100 time steps]", fontsize = 20)
        plt.title('spiking activity of excitatory population over time', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

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
        import datetime
        filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #save spike traces for plots
        np.save("data/fed/spiketimes_inp_self_sustained"+filename1+".npy", spikeTimes)

        #spikestimes_a = np.load("data/fed/spiketimes_inp_self_sustained.npy", allow_pickle=True)
        #sp_in = plotRaster(spikestimes_a); plt.show()

        np.save("data/fed/cg1Spikes_self_sustained"+filename1+".npy", cg1Spikes[0].data)

        #cgspk = np.load("data/fed/cg1Spikes_self_sustained.npy")
        #plot_smoothed_average_spikes(cgspk, 100)
        #plt.imshow(cgspk, cmap='gray'); plt.show()
       
        np.save("data/fed/cg1_0Voltage_self_sustained"+filename1+".npy", cg1_0Voltage[0].data)
        np.save("data/fed/meanFreqOutArray_self_sustained"+filename1+".npy", meanFreqOutArray[0])
        #meanFreqOutArray = np.load("data/fed/meanFreqOutArray_self_sustained20231129-150445.npy")
        #meanFreqOutArray = calculateMeanFreqOut(runTime, cg1Spikes[0], numECores, windowsize=30)

        #raise Exception

if __name__ == "__main__":        
    main()
