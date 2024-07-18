# Unsupervised Classification of Spike Patterns with the Loihi Neuromorphic Processor

This project explored the design and implementation of a spike-based attractor network capable of learning attractor dynamics and classifying its patterns on [Intel Loihi](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html). <br />
Find the paper [here](add link).

## Abstract
The exploration of brain-inspired, spike-based computation in electronic systems is underway with the aim of developing unconventional computing technologies that possess brain-like computing abilities. The Loihi neuromorphic processor provides a low-power, large-scale network of programmable silicon neurons for brain-inspired artificial intelligence applications. In this paper, we exploit the Loihi processors and a theory-guided methodology for enabling autonomous on-the-fly learning. 
Our method ensures an efficient and rapid selection of the network's hyperparameters, enabling the neuromorphic processor to generate attractor states through unsupervised learning in real-time.
These states are effective models of working memories and decision-making processes in the human brain. Specifically, we follow a fast design process, where we fine-tune network parameters using mean-field theory. Moreover, we measure the network's learning ability in terms of its error correction and pattern completion aptitude. Finally, we measured neuron cores' dynamic energy consumption of 3.23uJ/timestep during learning and 0.24uJ/timestep during the recall phase for four attractors composed of 512 excitatory neurons and 256 shared inhibitory neurons. This study showcases how large-scale, low-power digital neuromorphic processors can be quickly pre-programmed to enable the autonomous generation of attractor states. These attractors are fundamental computational primitives that theoretical analysis and experimental evidence indicate as versatile and reusable components suitable for a wide range of cognitive tasks.
## Table of Contents

- [Project Overview](#project-overview)
- [Installation and Setup](#installation)
- [Methods](#usage)
- [Results](#results)
- [Licence](#license)
- [Contact](#contact)

## Project Overview

Here, the experimental procedures are explained following the order of the sections in the [paper](add link). The software was developed using the NxNet package which is only compatible with the Loihi 1 platform. As of 2022, Intel released a new platform integration of the neuromorphic hardware, Loihi 2 and software, Lava. For further development, the use of Lava is suggested due to the lack of software support on the outdated development platform.

## Installation and Setup

The codes are based on the software package, NxSDK. To get the software and hardware access to Intel's neuromorphic platform, you need to join [INRC](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview). Once you get the access, follow the installation procedures for NxSDK 2.0 provided by Intel to proceed. To execute each script on vlab environment, go to nxsdk_src and run:

```bash
$ SLURM=1 python <yourscript.py>
```
## Methods

### M1. Theory-based design of an attractor network
To execute the script:
```bash
$ SLURM=1 python ETF.py
```
Find the image file "nxsdk_src/figures/ETF/ETF_weights.png" to see the Effective Transfer Function for each recurrent excitatory efficacy.

### M2. Confirmation of attractor formation
To execute the script:
```bash
$ SLURM=1 python self_sustained_activity.py
```
Find the image file "nxsdk_src/figures/self_sustained_activity/self_sustained_activity.png" to see the demonstration of self-sustained spiking activity.

## Results

### R1. Unsupervised learning of spike patterns
To execute the script:
```bash
$ SLURM=1 python STDP_multi_populations.py
```
Find the image files contained in "nxsdk_src/figures/multi_population_learning/" to observe the learning of attractor dynamics, evolution of recurrent excitatory efficacies in synaptic matrix and distribution. 

### R2. Demonstration of error correction/pattern completion
To execute the script:
```bash
$ SLURM=1 python error_correction.py
```
Find the image file "nxsdk_src/figures/error_correction/accuracy.png" to see the plot for the accuracy of attractor retrieval compared against hamming distance.

### R3. Unsupervised learning of overlapping/non-orthogonal spike patterns
To execute the script:
```bash
$ SLURM=1 python overlap.py
```
Find the image files contained in "nxsdk_src/figures/multi_population_learning/" to observe the learning of attractor dynamics, evolution of recurrent excitatory efficacies in synaptic matrix and distribution when the input spike patterns are overlapped. 

### R4. Energy profiling
To execute the script to measure the power consumption during inference:
```bash
$ SLURM=1 python energy_inference.py
```
Find the excel file "nxsdk_src/data/energy/energy_inference.png" to retrieve the raw data of power consumption. <br />

To execute the script to measure the power consumption during training:
```bash
$ SLURM=1 python energy_learning.py
```
Find the excel file "nxsdk_src/data/energy/energy_learning.png" to retrieve the raw data of power consumption.

[//]: <> (## Licence)

## Contact
Please feel free to send us inquiries about the codes to [Ryoga Matsuo](mailto:ryogaja7@gmail.com) [Federico Corradi](mailto:f.corradi@tue.nl)
