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
    cg1 = []
    numEperCore = 128
    cg1 = net.createCompartmentGroup(size=0) 
    Ecg = []
    for coreID in range(numECores//numEperCore):
        compartmentPrototype1 = nx.CompartmentPrototype(
            vThMant=180,         #vThMant*2^(6+vTHexp)
            functionalState=2,    #IDLE... Compartment gets serviced
            compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
            enableNoise = 1,
            refractoryDelay = 3,
            noiseMantAtRefractoryDelay = 1,
            noiseExpAtRefractoryDelay = 1,
            vMinExp = 0,
            enableSpikeBackprop=1,
            enableSpikeBackpropFromSelf=1,
            logicalCoreId = coreID,
            )
        Ecg.append(net.createCompartmentGroup(size=numEperCore, prototype=compartmentPrototype1))
        cg1.addCompartments(Ecg[coreID])
   

    # Create an inhibitory compartment group consisting of numICores neuron compartments
    # cg2... inhibitory population
    numIperCore = 64
    cg2 = net.createCompartmentGroup(size=0)
    for coreiID in range(numICores//numIperCore):
        compartmentPrototype2 = nx.CompartmentPrototype(
                vThMant=180,         #vThMant*2^(6+vTHexp)
                functionalState=2,    #IDLE... Compartment gets serviced
                compartmentVoltageDecay=int(1/(tau)*2**12), #int(1/tau*2**12)
                refractoryDelay = 3,
                vMinExp = 0,
                logicalCoreId = numECores//numEperCore+coreiID,
                )
        cg2.addCompartments(net.createCompartmentGroup(size=numIperCore, prototype=compartmentPrototype2))
   
            
    # Create an E-STDP learning rule used by the learning-enabled synapse 
    # 86 -> 30-36 
    # 54 -> 30-30
    # 40 -> 22-21
    lr = net.createLearningRule(dw='2^-3*x1*y0-2^-3*y1*x0-2^-4*sgn(w-23)*x1*y0-2^-4*x1*y0',
                                x1Impulse=x1Imp,
                                x1TimeConstant=x1Tau,
                                y1Impulse=y1Imp,
                                y1TimeConstant=y1Tau,
                                tEpoch=5)  

    # Set up the connection prototypes
    connProtoRecurrent = nx.ConnectionPrototype(weight=Jee,  #weight initialized at Jee
                                                enableLearning=1, 
                                                learningRule=lr,  
                                                delay=1,
                                                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                numWeightBits=6, #8
                                                weightLimitMant=2,
                                                weigthLimitExp=2,
                                                )     
    connProto1 = nx.ConnectionPrototype(weight=35, delay=5, numDelayBits=3,numWeightBits=6) 
    connProto2 = nx.ConnectionPrototype(weight=-15, delay=3, numDelayBits=3,numWeightBits=8)
    connProto3 = nx.ConnectionPrototype(weight=-15, delay=3, numDelayBits=3,numWeightBits=8)
    connProtoIn = nx.ConnectionPrototype(weight=45, delay=1, numDelayBits=3, numWeightBits=8) # 30 20 10
    connProtoIn2 = nx.ConnectionPrototype(weight=33, delay=5, numDelayBits=3,numWeightBits=6) # 15 10 5 
    connProtoReset = nx.ConnectionPrototype(weight=-80, delay=5, numWeightBits=4)
    connProtoNoise = nx.ConnectionPrototype(weight=5, delay=5, numWeightBits=6) 
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
   
    # Connect the excitatory population to the inhibitory population
    synaptic_connectivity3 = 0.3
    synaptic_matrix3 = np.random.rand(numICores, numECores) < synaptic_connectivity3
    cg1.connect(cg2, prototype=connProto1, connectionMask=synaptic_matrix3)  

    # Create the recurrent connections of the inhibitory population
    synaptic_connectivity4 = 0.45
    synaptic_matrix4 = np.random.rand(numICores, numICores) < synaptic_connectivity4
    for row in range(numICores):
        for col in range(numICores):
            if row == col:
                synaptic_matrix4[row][col] = 0
    cg2.connect(cg2, prototype=connProto2, connectionMask=synaptic_matrix4)
    
    # Connect the inhibitory population to the excitatory population
    synaptic_connectivity5 = 0.22
    synaptic_matrix5 = np.zeros((numECores, numICores)) < synaptic_connectivity5
    for row in range(numECores):
        row_matrix = np.random.rand(1,numICores) < synaptic_connectivity5
        synaptic_matrix5[row][:] = row_matrix
    cg2.connect(cg1, prototype=connProto2, connectionMask=synaptic_matrix5)          
    
    # Connect the learning recurrent synapses to the excitatory population
    synaptic_connectivity6 = 0.205#+ 0.001 #0.001 to compensate for the bits that are to be flipped in loops
    synaptic_matrix6 = np.bitwise_and(np.random.rand(numECores, numECores) < synaptic_connectivity6, \
                                      np.invert(np.eye(numECores, dtype=bool))*1)
    cg1.connect(cg1, prototype=connProtoRecurrent, connectionMask=synaptic_matrix6)

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
    VoltageProbe=[]
    for pop in range(numECores//numEperCore):
        VoltageProbe.append(cg1[pop*numEperCore].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE, probeConditions = None))
    for row in range(len(connectRecurrent)):
        for connection in range(len(connectRecurrent[row])):
            if connectRecurrent[row][connection]==0:
                # Fill no connections with 0
                weightProbe[row].append(0)
            else:
                weightProbe[row].append(connectRecurrent[row][connection].probe(nx.ProbeParameter.SYNAPSE_WEIGHT, probeConditions = vPc)) 
    return SpikesProbe, VoltageProbe, weightProbe

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
    initialIdleTime = 300
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
    binSize = 300
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
    runTime = 30300#90300#30260 
    print("The total number of timesteps is ", runTime, ".")
    # Spiking probability of the input spikes and noise 
    inputSpikingProbability = [0.05, 0.6] # [low frequency input, high frequency input]
    EnoiseSpikingProbability = 0.01
    InoiseSpikingProbability = 0.01
    #inputFraction = 3
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
    Jee = 0 
    # Learning parameters
    ImpParam = [127]#[10,20,30,40,41,42,43,44,45,46,47,47,47,47,48,49,50,44,44,44,44,44,44,50,50,50,50,50,50,51,51,60,60,50,50,10,10,10,20,20,23,25,26,27,12,90,10,1,2,3,4,5,6,7,8,9,15,55,64,33,32,22,24,66] #50
    x1Tau=40
    y1Tau=1
    
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
    
        if(runTime==90300):
            np.save("data/fed/stdp/spikeTimes.npy", spikeTimes)
            np.save("data/fed/stdp/spikeResetTimes.npy", spikeResetTimes)
            np.save("data/fed/stdp/spikeENoiseTimes.npy", spikeENoiseTimes)
            np.save("data/fed/stdp/spikeINoiseTimes.npy", spikeINoiseTimes)
        else:
            np.save("data/fed/stdp/spikeTimes"+str(runTime)+".npy", spikeTimes)
            np.save("data/fed/stdp/spikeResetTimes"+str(runTime)+".npy", spikeResetTimes)
            np.save("data/fed/stdp/spikeENoiseTimes"+str(runTime)+".npy", spikeENoiseTimes)
            np.save("data/fed/stdp/spikeINoiseTimes"+str(runTime)+".npy", spikeINoiseTimes)
    else:
        print("loading spikes from disk")
        if(runTime==90300):
            spikeTimes = np.load("data/fed/stdp/spikeTimes.npy", allow_pickle=True)
            spikeResetTimes = np.load("data/fed/stdp/spikeResetTimes.npy", allow_pickle=True)
            spikeENoiseTimes = np.load("data/fed/stdp/spikeENoiseTimes.npy", allow_pickle=True)
            spikeINoiseTimes = np.load("data/fed/stdp/spikeINoiseTimes.npy", allow_pickle=True)
        else:
            spikeTimes = np.load("data/fed/stdp/spikeTimes"+str(runTime)+".npy", allow_pickle=True)
            spikeResetTimes = np.load("data/fed/stdp/spikeResetTimes"+str(runTime)+".npy", allow_pickle=True)
            spikeENoiseTimes = np.load("data/fed/stdp/spikeENoiseTimes"+str(runTime)+".npy", allow_pickle=True)
            spikeINoiseTimes = np.load("data/fed/stdp/spikeINoiseTimes"+str(runTime)+".npy", allow_pickle=True)
 
    filelog = open("data/fed/stdp/logfiles.txt", "a") 
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
        (cg1Spikes, cg1_0Voltage, weightProbe) = setupNetwork(net, 
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
        sumWeight = np.zeros(runTime//dt)
        count = np.zeros(runTime//dt)

        for time in range(len(weightArray)):
            # Reshape the weight vectors into matrices
            weightMatrix.append(weightArray[time].reshape(numECores,numECores))
            for pop in range(numESubPop):
                weight_ave[time].append(np.nanmean(weightMatrix[time][pop*numESubCores:(pop+1)*numESubCores, \
                                                                pop*numESubCores:(pop+1)*numESubCores]))
        print("Time  | Avg Wgt. of Subpopulations")
        for time in range(len(weightArray)):
            string = str((time+1)*dt) + " | "
            for pop in range(numESubPop):
                string = string + str(round(weight_ave[time][pop])) + " | "
            print(str(string))

        #weight matrix
        import datetime
        filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save("data/fed/stdp/weightmatrix"+filename1+str(ImpParam[trial])+".npy", weightMatrix)
        #ww = np.load("data/fed/stdp/weightmatrix20231201-124620.npy")
        #save 
        np.save("data/fed/stdp/spiketimes"+filename1+str(ImpParam[trial])+".npy", spikeTimes)
        np.save("data/fed/stdp/cg1Spikes"+filename1+str(ImpParam[trial])+".npy", cg1Spikes[0].data)
        for pop in range(numECores//128):
            v1 = cg1_0Voltage[pop][0].data
            np.save("data/fed/stdp/pop_"+str(pop)+"fileid"+filename1+str(ImpParam[trial])+".npy", v1)

        print(filename1)
        filelog.write(filename1+" "+str(ImpParam[trial])+"\n")

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
    filelog.close()    
if __name__ == "__main__":        
    main()
