# Import modules
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
import pandas as pd

# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Set up the network architecture
def setupNetwork(net, numECores, numICores, tau, spikeGen, Enoise, Inoise, Jee):

    # Create an excitatory compartment group consisting of numECores neuron compartments
    # cg1... excitatory population
    compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            #axonDelay = 16, 
            enableNoise = 1,
            refractoryDelay = 6,#randomize,
            #randomizeRefractoryDelay = True,
            noiseMantAtRefractoryDelay = 7,
            noiseExpAtRefractoryDelay = 3,
            vMinExp = 0,
            enableSpikeBackprop=1,
            enableSpikeBackpropFromSelf=1,
            logicalCoreId = 0#coreID,
            )
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
   

    # Create an inhibitory compartment group consisting of numICores neuron compartments
    # cg2... inhibitory population
    compartmentPrototype2 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            enableNoise = 1,
            refractoryDelay = 6,#randomize,
            #randomizeRefractoryDelay = True,
            noiseMantAtRefractoryDelay = 7,
            noiseExpAtRefractoryDelay = 2,
            vMinExp = 0,
            logicalCoreId = 1#numECores//numEperCore+coreiID,
            )
    cg2 = net.createCompartmentGroup(size=numICores, prototype=compartmentPrototype2)
    connProtoRecurrent = nx.ConnectionPrototype(weight=Jee,  #weight initialized at Jee
                                                enableLearning=0,
                                                #learningRule=,
                                                delay=1,
                                                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                numWeightBits=4, #this was 8
                                                weightLimitMant=2,#4,
                                                weigthLimitExp=2,
                                                )


    # Set up the connection prototypes by specifying the weights (synaptic efficacies)
    #connProto1 = nx.ConnectionPrototype(weight=30, delay=5) 
    #connProto2 = nx.ConnectionPrototype(weight=-30, delay=5) 
    #connProtoNoise = nx.ConnectionPrototype(weight=10, delay=5) 
    connProto1 = nx.ConnectionPrototype(weight=35, delay=5,numDelayBits=3, numWeightBits=6)
    connProto2 = nx.ConnectionPrototype(weight=-15, delay=3,numDelayBits=3, numWeightBits=8)
    connProto3 = nx.ConnectionPrototype(weight=-15, delay=3,numDelayBits=3, numWeightBits=8)
    connProtoIn = nx.ConnectionPrototype(weight=48, delay=1,numDelayBits=3, numWeightBits=8)#48, numWeightBits=6) #6
    connProtoIn2 = nx.ConnectionPrototype(weight=33, delay=5,numDelayBits=3, numWeightBits=6)#6!, numWeightBits=4)#, numWeightBits=4) #6 #numDelayBits=2,numWeightBits=4
    #connProtoReset = nx.ConnectionPrototype(weight=-80, delay=5, numWeightBits=4)
    #connProtoRecurrentFixed = nx.ConnectionPrototype(weight=21, delay=5)
    connProtoNoise = nx.ConnectionPrototype(weight=15, delay=5, numWeightBits=6)


    # Create the recurrent connections of the excitatory population
    synaptic_connectivity1 = 0.22
    synaptic_matrix1 = np.random.rand(numECores, numECores) < synaptic_connectivity1
    spikeGen.connect(cg1, prototype=connProtoRecurrent, connectionMask=synaptic_matrix1) 
    
    # Connect noise sources to neurons
    Enoise.connect(cg1, prototype=connProtoNoise, connectionMask=np.eye(numECores)) 
    Inoise.connect(cg2, prototype=connProtoNoise, connectionMask=np.eye(numICores)) 

    # Create the connections from the excitatory population to the inhibitory population
    synaptic_connectivity2 = 0.4
    synaptic_matrix2 = np.random.rand(numICores, numECores) < synaptic_connectivity2
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix2) 
    
    # Create the recurrent connections of the inhibitory population
    synaptic_connectivity3 = 0.50
    synaptic_matrix3 = np.random.rand(numICores, numICores) < synaptic_connectivity3
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix3)
    
    # Create the connections from the inhibitory population to the excitatory population
    synaptic_connectivity4 = 0.11
    synaptic_matrix4 = np.random.rand(numECores, numICores) < synaptic_connectivity4
    cg2.connect(cg1, prototype=connProto3, connectionMask=synaptic_matrix4)

    # Configure probes
    probeParameters = nx.ProbeParameter.SPIKE
    SpikesProbeExc  = cg1.probe(probeParameters, probeConditions = None)
    SpikesProbeInh  = cg2.probe(probeParameters, probeConditions = None)

    # Return probes for post-synaptic spikes in the excitatory and inhibitory populations
    return SpikesProbeExc, SpikesProbeInh

# Create numECores=numSpikeGen spike generators 
def generateSpikes(net, runTime, numSpikeGen, spikingProbability):    
    spikeTimes = [[] for _ in range(numSpikeGen)]
    lenProb = len(spikingProbability)
    offset = runTime//lenProb
    for prob in range(lenProb):
        spikeTimesProb = genPoissonSpikeTimes(offset, spikingProbability[prob], numSpikeGen)
        for port in range(numSpikeGen):
            for time in range(len(spikeTimesProb[port])):
                spikeTimesProb[port][time] = spikeTimesProb[port][time] + offset*prob 
            spikeTimes[port] = spikeTimes[port] + spikeTimesProb[port] 
    spikeGen = createSpikeGenerators(net, numSpikeGen, spikeTimes)
       
    return spikeGen, spikeTimes

# Produce spike timing of numSpikeGen spike generators
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

# Create numSpikeGen spike generators
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   #generate spikes in numSpikeGen ports
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) #addSpikes(portID, spikeTimes)
    return spikeGen

# Calculate the mean spiking frequency of the input generators
def calculateMeanFreqIn(spikeTimes, numCores, runTime, length):
    sumFreq = np.zeros(length)
    aveFreq = np.zeros(length)
    Freq = np.zeros((numCores, length))
    period = runTime//length
    idx = 0
    for neuron in range(numCores):
        spikeTimeArray = np.array(spikeTimes[neuron])
        for idx in range(length):
            spikeBool = np.where(np.logical_and(spikeTimeArray >= idx*period, \
                                                   spikeTimeArray < (idx+1)*period), 1, 0)
            Freq[neuron, idx] = np.count_nonzero(spikeBool)*(100/period)
            sumFreq[idx] += Freq[neuron, idx]
    
    for idx in range(length):
        aveFreq[idx] = sumFreq[idx]/numCores
            
    return aveFreq    
         

# Calculate the mean post-synaptic spiking frequency of the excitatory population
def calculateMeanFreqOut(postSynapticSpikes, numCores, runTime, length):
    sumFreq = np.zeros(length)
    sumVar = np.zeros(length)
    aveFreq = np.zeros(length)
    variance = np.zeros(length)
    spikeTimes = []
    Freq = np.zeros((numCores, length))
    period = runTime//length
    idx = 0
    for neuron in range(numCores):
        spikeTimes.append(np.where(postSynapticSpikes.data[neuron])[0].tolist())
        spikeTimeArray = np.array(spikeTimes[neuron])
        for idx in range(length):
            spikeBool = np.where(np.logical_and(spikeTimeArray >= (idx+0.5)*period, \
                                                   spikeTimeArray < (idx+1)*period), 1, 0)
            Freq[neuron, idx] = np.count_nonzero(spikeBool)*(100*2/period)
            sumFreq[idx] += Freq[neuron, idx]
    
    for idx in range(length):
        aveFreq[idx] = sumFreq[idx]/numCores
            
    for neuron in range(numCores):
        for idx in range(length):
            sumVar[idx] += (Freq[neuron, idx] - aveFreq[idx])**2
    for idx in range(length):        
        variance[idx] = sumVar[idx]/numCores
            
    return aveFreq, variance                 
        
        
if __name__ == "__main__":        
    # Time duration for the execution
    runTime = 60000
    
    # Number of excitatory neurons
    numECores = 128
    # Number of inhibitory neurons
    numICores = 64

    # Refractory period of the input spikes
    tEpoch = 1

    # An array of spiking probabilities of the input spike generators
    spikingProbability = [(0.4*n**2) for n in np.arange(0, 1, 0.03)]

    # Spiking probabilitoes of the noise generators
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1

    # Number of different frequencies of the input spikes
    numProb = len(spikingProbability)
    
    # Voltage decay time constant
    tau = 16
   
    # Recurrent efficacies of the excitatory connections for which the ETF is measured for
    Jee = [15, 21, 30, 35]
    len_Jee = len(Jee)
   
    # Variable initialization
    cg1Spikes = [[[] for _ in range(numProb)] for _ in range(len_Jee)]
    meanFreqInArray = np.zeros((numProb, len_Jee))
    meanFreqOutArray = np.zeros((numProb, len_Jee))
    meanFreqOutArrayInh = np.zeros((numProb, len_Jee))
    variance_out = np.zeros((numProb, len_Jee))
    variance_outInh = np.zeros((numProb, len_Jee))
    legend_labels = []
    
    # Create arrays containing the spike times of the noise generators from the predefined spiking probabilities
    spikeENoiseTimes = genPoissonSpikeTimes(runTime, EnoiseSpikingProbability, numECores) 
    spikeINoiseTimes = genPoissonSpikeTimes(runTime, InoiseSpikingProbability, numICores) 
    
    for ii in range(len_Jee):
        # Initialize a network
        net = nx.NxNet()

        ### Configure the network
        # Create arrays containing spike generator objects and spike times of these generators
        (spikeGen, spikeTimes) = generateSpikes(net, 
                                                runTime, 
                                                numECores, 
                                                spikingProbability)
        
        # Create noise generator objects
        Enoise = createSpikeGenerators(net, numECores, spikeENoiseTimes) 
        Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes)    

        # Create neurons and construct the network
        (cg1Spikes, cg1SpikesInh) = setupNetwork(net, 
                                                 numECores, 
                                                 numICores, 
                                                 tau, 
                                                 spikeGen,
                                                 Enoise,
                                                 Inoise,
                                                 Jee[ii])
        #------------------------------------------------------------------------
        # Run    
        #------------------------------------------------------------------------
        net.run(runTime, executionOptions={'partition':'nahuku32'})
        net.disconnect()
        #------------------------------------------------------------------------
        # Data Processing
        #------------------------------------------------------------------------
        # Calculates the mean input frequencies for different spiking probabilities
        meanFreqInArray[:, ii] = calculateMeanFreqIn(spikeTimes, 
                                                     numECores, 
                                                     runTime, 
                                                     len(spikingProbability))
        # Calculates the mean output (post-synaptic) frequencies for different input frequencies
        (meanFreqOutArray[:, ii] , variance_out[:, ii] ) =\
                                calculateMeanFreqOut(cg1Spikes[0], 
                                                     numECores, 
                                                     runTime,
                                                     len(spikingProbability)) 
        print(meanFreqOutArray[:, ii])
        (meanFreqOutArrayInh[:, ii], variance_outInh[:, ii] ) =\
                                calculateMeanFreqOut(cg1SpikesInh[0], 
                                                     numICores, 
                                                     runTime,
                                                     len(spikingProbability))      
        legend_labels.append(r"$J_{E,E} =$" + str(Jee[ii]))


    # Plot
    fig1 = plt.figure(1001, figsize=(25,25))
    plt.xlabel('mean presynaptic spiking frequency [/100 timesteps]', fontsize = 30)
    plt.ylabel('mean postsynaptic spiking frequency [/100 timesteps]', fontsize = 30)
    plt.title('Effective Transfer Function', fontsize = 30)

    plt.tick_params(axis='both', which='major', labelsize=20)
    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for ii in range(len(Jee)):
        # Effective TF 
        plt.plot(meanFreqInArray[:, ii], meanFreqOutArray[:, ii], color=plot_color[ii])
        # Standard deviation
        plt.errorbar(meanFreqInArray[:, ii], 
                     meanFreqOutArray[:, ii], 
                     yerr=np.sqrt(variance_out[:, ii]), 
                     fmt='o', 
                     capsize=10, 
                     color='black', 
                     ecolor= plot_color[ii])
    plt.legend(legend_labels, fontsize = 20)
    # Stability line
    #plt.plot(list(range(0, 30)), list(range(0,30)), color = plot_color[ii+1])
    #plt.xlim(0, 30)
    #plt.ylim(0, 30)

    # Save the measurement data to dataframe
    data = {}
    for ii in range(len(Jee)):
        data['FreqIn'+str(ii)] =  meanFreqInArray[:, ii]
        data['FreqOutExc'+str(ii)] =  meanFreqOutArray[:, ii]
        data['Var'+str(ii)] =  variance_out[:, ii]
        data['FreqOutInh'+str(ii)] =  meanFreqOutArrayInh[:, ii]

    df = pd.DataFrame(data) 
    import datetime
    filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(filename1)
    df.to_excel('data/fed/ETF/ETF_weights'+filename1+'.xlsx', index=False)
    
    # Save the figure
    if haveDisplay:
        plt.show()
    else:
        fileName = "data/fed/ETF/ETF_weights"+filename1+".png"
        print("No display available, saving to file " + fileName + ".")
        fig1.savefig(fileName)
