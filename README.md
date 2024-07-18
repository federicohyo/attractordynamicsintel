# Unsupervised Classification of Spike Patterns with the Loihi Neuromorphic Processor

This project explored the design and implementation of a spike-based attractor network capable of learning attractor dynamics and classifying its patterns on [Intel Loihi](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html). <br />
Find the paper [here](add link).

## Paper

Ryoga Matsuo, Ahmed Elgaradiny, and Federico Corradi "Unsupervised Classification of Spike Patterns with the Loihi Neuromorphic Processor", under-review, 2024


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
