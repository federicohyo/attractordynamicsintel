# Import modules

import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import os
import numpy as np
import matplotlib as mpl
from nxsdk.utils.plotutils import plotRaster
from nxsdk.graph.monitor.probes import *
import pandas as pd
import matplotlib.colors as colors
import pickle

# For image file display
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

# Set up the network architecture
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
    # Create an excitatory compartment group consisting of numECores neuron compartments
    # cg1... excitatory population
    # Ecg... list of neurons to be used for probing
    #cg1 = []
    numEperCore = 128
    #cg1 = net.createCompartmentGroup(size=0) 
    #Ecg = []
    #for coreID in range(numECores//numEperCore):
    compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            ####axonDelay = 16, 
            ##thresholdBehavior = 1,
            ##enableNoise = 1,
            refractoryDelay = 3, #randomize,
            ####randomizeRefractoryDelay = True,
            ##noiseMantAtRefractoryDelay = 5, 
            ##noiseExpAtRefractoryDelay = 2,
            vMinExp = 0,
            enableSpikeBackprop=1,
            enableSpikeBackpropFromSelf=1,
            logicalCoreId = 0#coreID,
            )
    #    Ecg.append(net.createCompartmentGroup(size=numEperCore, prototype=compartmentPrototype1))
    #    cg1.addCompartments(Ecg[coreID])
    cg1 = net.createCompartmentGroup(size=numECores, prototype=compartmentPrototype1)
   

    # Create an inhibitory compartment group consisting of numICores neuron compartments
    # cg2... inhibitory population
    numIperCore = 64
    #cg2 = net.createCompartmentGroup(size=0)
    #for coreiID in range(numICores//numIperCore):
    compartmentPrototype2 = nx.CompartmentPrototype(
                vThMant=180,         #vThMant*2^(6+vTHexp)
                functionalState=2,    #IDLE... Compartment gets serviced
                compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
                enableNoise = 0,
                refractoryDelay = 3,
                #refractoryDelay = 6,#randomize,
                #thresholdBehavior = 1,  
                #noiseMantAtRefractoryDelay = 5, 
                #noiseExpAtRefractoryDelay = 2, 
                vMinExp = 0,
                logicalCoreId = 1#numECores//numEperCore+coreiID,
                )
    #    cg2.addCompartments(net.createCompartmentGroup(size=numIperCore, prototype=compartmentPrototype2))
    cg2 = net.createCompartmentGroup(size=numICores, prototype=compartmentPrototype2)
   
            
    # Create an E-STDP learning rule used by the learning-enabled synapse 
    # 86 -> 30-36 
    # 54 -> 30-30
    # 40 -> 22-21
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-16)*x1*y0-2^-4*x1*y0',
    #lr = net.createLearningRule(dw='2^-5*x1*y0-2^-5*y1*x0-2^-5*sgn(w-50)*x1*y0-2^-5*x1*y0',
    #lr = net.createLearningRule(dw='2^-2*x1*y0-2^-2*y1*x0-2^-2*sgn(w-50)*x1*y0-2^-2*x1*y0', 125! 5  - 26
    #lr = net.createLearningRule(dw='2^-2*x1*y0-2^-5*y1*x0-2^-2*sgn(w-50)*x1*y0-2^-2*x1*y0', 125! 16 - 42
    #lr = net.createLearningRule(dw='2^-2*x1*y0-2^-5*y1*x0-2^-2*sgn(w-50)*x1*y0-2^-2*x1*y0', 125! 19 - 45
    #lr = net.createLearningRule(dw='2^-2*x1*y0-2^-2*y1*x0-2^-2*sgn(w-50)*x1*y0-2^-2*x1*y0', 125! 18 - 45
    #lr = net.createLearningRule(dw='2^-2*x1*y0-2^-4*y1*x0-2^-2*sgn(w-50)*x1*y0-2^-2*x1*y0', 125! 11 - 36 
    #lr = net.createLearningRule(dw='2^-4*x1*y0-2^-4*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-2*x1*y0', 10! 0 - 7 
    #lr = net.createLearningRule(dw='2^-5*x1*y0-2^-5*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-2*x1*y0', 125! 0 - 7
    #lr = net.createLearningRule(dw='2^-5*x1*y0-2^-5*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-5*x1*y0', 125! 14 - 40
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-5*x1*y0', 125! 19 - 46 
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-4*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-2*x1*y0', 125! 2 - 16
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-4*y1*x0-2^-2*sgn(w-21)*x1*y0-2^-2*x1*y0',  20! 8 - 10 OK SUB ATTRACTOR
    # 24 in eq 4 makes it 12 instead of 8
    #lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-23)*x1*y0-2^-4*x1*y0', 
    lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-23)*x1*y0-2^-4*x1*y0', 
            # 19 for perfect dynamic starting point jee 21
            #next would be to make it 15
                                x1Impulse=x1Imp,
                                x1TimeConstant=x1Tau,
                                y1Impulse=y1Imp,
                                y1TimeConstant=y1Tau,
                                tEpoch=5) #tEpoch 5 

    # Set up the connection prototypes
    connProtoRecurrent = nx.ConnectionPrototype(weight=Jee,  #weight initialized at Jee
                                                enableLearning=1,#1, 
                                                learningRule=lr,  
                                                delay=1,
                                                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                numWeightBits=6, #this was 8
                                                weightLimitMant=2,#4,
                                                weigthLimitExp=2,
                                                )     
    connProto1 = nx.ConnectionPrototype(weight=35, delay=5,numDelayBits=3, numWeightBits=6)
    connProto2 = nx.ConnectionPrototype(weight=-15, delay=5,numDelayBits=3, numWeightBits=8)
    connProto3 = nx.ConnectionPrototype(weight=-15, delay=5,numDelayBits=3, numWeightBits=8)
    connProtoIn = nx.ConnectionPrototype(weight=48, delay=5,numDelayBits=3, numWeightBits=8)#48, numWeightBits=6) #6
    connProtoIn2 = nx.ConnectionPrototype(weight=33, delay=5,numDelayBits=3, numWeightBits=6)#6!, numWeightBits=4)#, numWeightBits=4) #6 #numDelayBits=2,numWeightBits=4
    connProtoReset = nx.ConnectionPrototype(weight=-80, delay=5, numWeightBits=4)
    #connProtoRecurrentFixed = nx.ConnectionPrototype(weight=21, delay=5)
    connProtoNoise = nx.ConnectionPrototype(weight=10, delay=5, numWeightBits=6)

    print(numECores)
    print(numICores)
    # Connect each generator to each excitatory neuron
    synaptic_matrix0 = np.eye(numECores)
    Epc1.connect(cg1, prototype=connProtoIn, connectionMask=synaptic_matrix0) 
        
    # Connect generators to inhibitory pool
    synaptic_matrix1 = np.concatenate((np.eye(numICores), np.eye(numICores)), axis = 1)
    Epc1.connect(cg2, prototype=connProtoIn2, connectionMask=synaptic_matrix1) 
        
    # Connect Ireset and the inhibitory population (reset of the excitatory neuronal activity)
    Ireset.connect(cg1, prototype=connProtoReset)
    
    # Connect noise source to neurons
    Enoise.connect(cg1, prototype=connProtoNoise, connectionMask=np.eye(numECores))
    Inoise.connect(cg2, prototype=connProtoNoise, connectionMask=np.eye(numICores))
 
    # Connect the learning recurrent synapses to the excitatory population
    # 0.218 also ok
    synaptic_connectivity6 = 0.208#5#0.35#2# 0.4 there is attractor #26 #+ 0.001 #0.001 to compensate for the bits that are to be flipped in loops
    synaptic_matrix6 = np.bitwise_and(np.random.rand(numECores, numECores) < synaptic_connectivity6, \
                                      np.invert(np.eye(numECores, dtype=bool))*1)
    #cg1.connect(cg1, prototype=connProtoRecurrent, connectionMask=synaptic_matrix6)
    #cg1.connect(cg1, prototype=connProtoRecurrentFixed, connectionMask=synaptic_matrix6)
  
    # Connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.3#0.35
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3)  

    # Create the recurrent connections of the inhibitory population
    synaptic_connectivity4 = 0.45#0.5
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    # Connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.22#0.35#19 #192 #16
    synaptic_matrix5 = np.random.rand(numECores, numICores) < synaptic_connectivity5
    cg2.connect(cg1, prototype=connProto3, connectionMask=synaptic_matrix5)          
    
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

    # Configure probes
    vPc = nx.IntervalProbeCondition(dt=dt, tStart=dt-1)
    weightProbe = [[] for _ in range(len(connectRecurrent))]
    SpikesProbe = cg1.probe(nx.ProbeParameter.SPIKE, probeConditions = None)
    SpikesProbeIn  = cg2.probe(nx.ProbeParameter.SPIKE, probeConditions = None)
    VoltageProbeIn = cg2[0].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None)
    VoltageProbe=[]
    for pop in range(numECores//numEperCore):
        VoltageProbe.append(cg1[16].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None))
    #cg1[pop*numEperCore].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None))
    for row in range(len(connectRecurrent)):
        for connection in range(len(connectRecurrent[row])):
            if connectRecurrent[row][connection]==0:
                # Fill no connections with 0
                weightProbe[row].append(0)
            else:
                weightProbe[row].append(connectRecurrent[row][connection].probe(nx.ProbeParameter.SYNAPSE_WEIGHT, probeConditions = vPc)) 
    return SpikesProbe, VoltageProbe, weightProbe, SpikesProbeIn, VoltageProbeIn

# Create numSpikeGen spike generators
def createSpikeGenerators(net, numSpikeGen, spikeTimes):
    spikeGen = net.createSpikeGenProcess(numSpikeGen)   
    for spikeGenPort in range(0, numSpikeGen):
        spikeGen.addSpikes(spikeGenPort, spikeTimes[:][spikeGenPort]) 
    return spikeGen

# Produce spike times of numSpikeGen spike generators
def genInputSpikeTimes(runTime, spikingProbability, tEpoch, numSpikeGen, numESubPop, repetition):
    runTimeRep = runTime//repetition
    spikeTimes = []
    initialIdleTime = 500
    spikes = np.zeros((numSpikeGen, runTime)) 
    numNeuronsPerSubpop = round(numSpikeGen/numESubPop)
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

# Produce spike times for the resetting spikes
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

# Produce spike times of the noise sources
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

# Calculate the mean post-synaptic spiking frequency of the excitatory population over time for each subpopulation
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
    
def main():
    
    # Time duration of the execution
    runTime = 15280#80100#15280#30185#270900#180600#90300#17100#90300
    print("The total number of timesteps is ", runTime, ".")
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = [0.05, 0.6] # [low frequency input, high frequency input]
    EnoiseSpikingProbability = 0.1
    InoiseSpikingProbability = 0.1
    #inputFraction = 1 
    # Number of neurons in each excitatory subpopulation
    numESubCores = 128
    # Number of excitatory subpopulations
    numESubPop = 1 
    # Number of excitatory neurons
    numECores = 128#numESubCores*numESubPop
    # Number of inhibitory neurons
    numICores = 64#numECores//2
    # Number of pixels
    numSpikeGen = numECores#numECores
    # Time constant
    tau=16 #16
    # Refractory period of the input spikes
    tEpoch = 1
    repetition = (runTime-300)//3000 
    # Initial recurrent weight (learning)
    Jee = 0#21
    # Learning parameters
    ImpParam = [127]#[25]
    x1Tau=85
    y1Tau=2
    #if starting point 21 -> 25,30,10
    #if starting point 0  -> 127,85,2
    
    # Sample cycle of weights
    dt = runTime//4
    
    # Generate spike times
    if(False):
        print("generating spikes...")
        spikeTimes = genInputSpikeTimes(runTime, 
                                      inputSpikingProbability, 
                                      tEpoch, 
                                      numSpikeGen, 
                                      numESubPop, 
                                      repetition)
        spikeResetTimes = genResetSpikeTimes(runTime, 32, numESubPop, repetition)
        spikeENoiseTimes = genNoiseSpikeTimes(runTime, EnoiseSpikingProbability, numECores)
        spikeINoiseTimes = genNoiseSpikeTimes(runTime, InoiseSpikingProbability, numICores)   
    
        if(runTime==90300 and numESubPop==4):
            np.save("data/fed/stdp/spikeTimes.npy", spikeTimes)
            np.save("data/fed/stdp/spikeResetTimes.npy", spikeResetTimes)
            np.save("data/fed/stdp/spikeENoiseTimes.npy", spikeENoiseTimes)
            np.save("data/fed/stdp/spikeINoiseTimes.npy", spikeINoiseTimes)
        else:
            np.save("data/fed/stdp/spikeTimes"+str(runTime)+str(numESubPop)+".npy", spikeTimes)
            np.save("data/fed/stdp/spikeResetTimes"+str(runTime)+str(numESubPop)+".npy", spikeResetTimes)
            np.save("data/fed/stdp/spikeENoiseTimes"+str(runTime)+str(numESubPop)+".npy", spikeENoiseTimes)
            np.save("data/fed/stdp/spikeINoiseTimes"+str(runTime)+str(numESubPop)+".npy", spikeINoiseTimes)
    else:
        print("loading spikes from disk")
        if(runTime==90300 and numESubPop==4):
            spikeTimes = np.load("data/fed/stdp/spikeTimes.npy", allow_pickle=True)
            spikeResetTimes = np.load("data/fed/stdp/spikeResetTimes.npy", allow_pickle=True)
            spikeENoiseTimes = np.load("data/fed/stdp/spikeENoiseTimes.npy", allow_pickle=True)
            spikeINoiseTimes = np.load("data/fed/stdp/spikeINoiseTimes.npy", allow_pickle=True)
        else:
            spikeTimes = np.load("data/fed/stdp/spikeTimes"+str(runTime)+str(numESubPop)+".npy", allow_pickle=True)
            spikeResetTimes = np.load("data/fed/stdp/spikeResetTimes"+str(runTime)+str(numESubPop)+".npy", allow_pickle=True)
            spikeENoiseTimes = np.load("data/fed/stdp/spikeENoiseTimes"+str(runTime)+str(numESubPop)+".npy", allow_pickle=True)
            spikeINoiseTimes = np.load("data/fed/stdp/spikeINoiseTimes"+str(runTime)+str(numESubPop)+".npy", allow_pickle=True)
 
        
    for trial in range(len(ImpParam)):
        # Initialize a network
        net = nx.NxNet()

        # Learning parameters
        x1Imp=ImpParam[trial]
        y1Imp=ImpParam[trial]
        
        # Create generators
        Epc1 = createSpikeGenerators(net, numSpikeGen, spikeTimes) 
        Ireset = createSpikeGenerators(net, 32, spikeResetTimes)
        Enoise = createSpikeGenerators(net, numSpikeGen, spikeENoiseTimes) 
        Inoise = createSpikeGenerators(net, numICores, spikeINoiseTimes) 
        print("Spikes were generated/or loaded. Configuring network...")    

        # Configure the network
        (cg1Spikes, cg1_0Voltage, weightProbe, cg2Spikes, cg2_0VoltageIn)= setupNetwork(net, 
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

        # Execute
        net.run(runTime, executionOptions={'partition':'nahuku08'})
        net.disconnect()
        print("Execution finished. Plotting synaptic matrices...")

        # Process and plot weights over time
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
        weight_std = [[] for _ in range(len(weightArray))]
        sumWeight = np.zeros(runTime//dt)
        count = np.zeros(runTime//dt)

        for time in range(len(weightArray)):
            # Reshape the weight vectors into matrices
            weightMatrix.append(weightArray[time].reshape(numECores,numECores))
            for pop in range(numESubPop):
                weight_ave[time].append(np.nanmean(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))
                weight_std[time].append(np.nanstd(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))

        print("Time  | Avg | Std | Wgt. of Subpopulations")
        for time in range(len(weightArray)):
            string = str((time+1)*dt) + " | "
            for pop in range(numESubPop):
                string = string + str(round(weight_ave[time][pop])) + " | "
                string = string + str(round(weight_std[time][pop])) + " | "
            print(str(string))

        #weight matrix
        import datetime
        filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save("data/fed/stdp/weightmatrix"+filename1+str(ImpParam[trial])+".npy", weightMatrix)
        #ww = np.load("data/fed/stdp/weightmatrix20231201-124620.npy")
        #save 
        np.save("data/fed/stdp/spiketimes"+filename1+str(ImpParam[trial])+".npy", spikeTimes)
        np.save("data/fed/stdp/cg1Spikes"+filename1+str(ImpParam[trial])+".npy", cg1Spikes[0].data)
        np.save("data/fed/stdp/cg2Spikes"+filename1+str(ImpParam[trial])+".npy", cg2Spikes[0].data)
        np.save("data/fed/stdp/pop_in_"+str(pop)+"fileid"+filename1+str(ImpParam[trial])+".npy", cg2_0VoltageIn[0].data)

        for pop in range(numECores//numECores):
            v1 = cg1_0Voltage[pop][0].data
            np.save("data/fed/stdp/pop_"+str(pop)+"fileid"+filename1+str(ImpParam[trial])+".npy", v1)

        print(filename1)

        if False:
            fig2 = plt.figure(1003, figsize=(40, 40))    
            #cmap = colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'black'])
            for time in range(4):
                ax = plt.subplot(2, 2, time+1)
                # Create a heatmap using matplotlib
                plt.imshow(weightMatrix[time], interpolation='nearest', vmin=0, vmax=80) 
                        #cmap=cmap, interpolation='nearest', vmin=0, vmax=80)
                cbar = plt.colorbar(shrink=0.9)
                cbar.ax.tick_params(labelsize=45)
                plt.title("t="+str((time+1)*dt), fontsize=60)
                plt.xticks(fontsize=45)
                plt.yticks(fontsize=45)

            if haveDisplay:
                plt.show()
            else:
                fileName = "figures/multi_population_learning/SynapticMatrices" + str(trial+1) + ".png"
                print("No display available, saving to file " + fileName + ".")
                fig2.savefig(fileName)



            # Plot probability distributions of the synaptic weights over time
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
                fileName = "figures/multi_population_learning/SynapticMatrices_Distribution" + str(trial+1) + ".png"
                print("No display available, saving to file " + fileName + ".")
                fig.savefig(fileName)

            
            # Process and plot pre,  postsynaptic spikes, membrane voltage and excitatory frequency over time
            fig = plt.figure(1001, figsize=(100, 70))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']

            ax0 = plt.subplot(4, 1, 1)
            plotRaster(spikeTimes)
            plt.title('presynaptic spikes', fontsize = 100)
            plt.xticks(fontsize=60)
            plt.yticks(fontsize=60)
            plt.ylabel("neuron index", fontsize = 80)

            ax1 = plt.subplot(4, 1, 2)
            cg1Spikes[0].plot()
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
                fileName = "figures/multi_population_learning/PSTH" + str(trial+1) + ".png"
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

            #save
            np.save("data/fed/meanfreqoutarray"+filename1+".npy", meanFreqOutArray)

            if haveDisplay:
                plt.show()
            else:
                fileName = "figures/multi_population_learning/PSTH" + str(trial+1) + ".png"
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
                    fileName = "figures/multi_population_learning/PSTH_Zoom"+str(ii)+ "_" + str(trial+1) + ".png"
                    print("No display available, saving to file " + fileName + ".")
                    fig.savefig(fileName)

            # Save the weight matrix data
            with open('data/data_weight_multi' + str(trial+1) +'.pkl', 'wb') as f:
                pickle.dump(weightMatrix , f) 
            print("Weights were saved.")
        
if __name__ == "__main__":        
    main()
