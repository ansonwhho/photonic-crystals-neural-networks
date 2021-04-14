# Design of photonic crystal waveguides using neural networks
Repository for my undergraduate dissertation, *Design of photonic crystal waveguides using neural networks*. 

**This repo is a work in progress**  
**Updated 2021-04-12**

# Motivation
The ability to guide slow light has many potential applications, such as in optical storage and the study of nonlinear optical effects. One of the more readily accessible ways of slowing light down is to propagate it through a photonic crystal, which through Bragg reflection is able to significantly reduce the group velocity of light. 

Waveguides designed using photonic crystals are thus very promising for manipulating light, but the practical utility of such a device is heavily limited by figures of merit, such as operational bandwidth and propagation loss. The aim is thus to design a photonic crystal waveguide (PhCW) that optimises both for a slow group velocity of light, as well as other figures of merit. These are: 
- Group index (n<sub>g0</sub>)
- Bandwidth
- Group-index bandwidth product (GBP)  
- Average loss
- Loss at n<sub>g0</sub>
- Delay

The seed design for the waveguide is the W1 line defect waveguide, and the design is altered using the following parameters: 
- p<sub>i</sub>, i = 1, 2, 3: Shift of the i<sup>th</sup> row of holes parallel to the waveguide  
- s<sub>i</sub>, i = 1, 2, 3: Shift of the i<sup>th</sup> row of holes perpendicular to the waveguide
- r<sub>i</sub>, i = 1, 2, 3: Radii of the i<sup>th</sup> row of holes
- r<sub>0</sub>: radii of the remaining holes in the waveguide

Trying to optimise these can be very challenging in practice, since small changes in waveguide design can lead to large (and unpredictable) changes in the output figures of merit. This is thus a high-dimensional problem that is difficult to solve using physical intuition, and also computationally expensive using numerical methods.

This project thus focusses on using neural networks to tackle this challenge of waveguide design. 

# Contents
## File structure
- `README.md`  
- [**backend**](###backend) 
  - `constraintsFix.py` (initially `constraints.py`)  
  - `experiment.py`  
  - `mpbParser.py`  
- [**models**](###models)
  - **designs**
    - **candidates**
      - `prelim-design-candidates.txt`
      - `final-candidates.txt`
      - `candidate-tester.py`
  - **predictions**  
  - `train_2021-04-11_v1.h5`
  - `train_2021-04-11_v2.h5`
- [**neural-networks**](###neural-networks)  
  - `architectures.py`
  - `design.py`
  - `predict.py`
  - `preprocessing.py`
  - `train.py`
- [**predict-sets**](###predict-sets)
- [**training-sets**](###training-sets)
  - **combined-sets**
  - **run-sets**
  - `train-set-generator.py`
- [**WaveguideCTL**](###WaveguideCTL)
  - `W1_2D_v04.ctl`
  - `W1_3D_v1.ctl`

## Description of contents
### backend
Contains Python files for accessing and obtaining the calculations from [MIT Photonic Bands](https://github.com/NanoComp/mpb) (MPB). [These were originally written by Sean Billings](https://github.com/sschulz365/PhC_Optimization). I made some edits to them (e.g. replacing `constraints.py` with `constraintsFix.py`) so that they would be compatible with Python 3 and to obtain the desired FOMs from the MPB calculations.
- `constraintsFix.py`: sets physical constraints on the possible waveguide design parameters
- `experiment.py`: for performing an MPB experiment
- `mpbParser.py`: parses the output from MPB to obtain a dictionary of the [six figures of merit](#Motivation)

### models
- **models**: for trained PhCW models and design candidates
  - **candidates**: candidates for PhCW design made using `design.py`
    - `prelim-design-candidates.txt`
    - `final-candidates.txt`
    - `candidate-tester.py`
  - `train_2021-04-11_v1.h5`: preliminary test model trained with 10 epochs
  - `train_2021-04-11_v2.h5`: trained model, 1000 epochs

### neural-networks
- `architectures.py`: for creating the neural network architecture and for changing the network hyperparameters. This project only uses a Keras sequential model, with emphasis being placed on the feasibility of such an approach (i.e. using neural networks) to designing PhCWs. 
- `design.py`: finds the most promising PhCW designs from predictions, to create candidate designs (see [**models**](###models))
- `predict.py`: for making predictions on a set of input parameters
- `preprocessing.py`: for data processing (e.g. removing invalid experiments) and exploratory data analysis
- `train.py`: for creating trained neural networks

### predict-sets
*Input sets for making predictions on*  
Contains CSVs with generated input parameters for testing - these were generated using `predict.py`, which can be found in [**neural-networks**](###neural-networks). 
- **predictions**: *Predictions made based on the input sets*. Contains predictions based on the above CSVs, for desired figures of merit. Predictions were made using `predict.py` and one of the models in [**models**](###models).

### training-sets
Calculations from the 2D MPB simulation, run using `W1_2D_v04.ctl` and either `train-set-generator.py` or `candidate-tester.py`. Used as training data for the models in [**models**](###models), and models are made using `train.py`. 

### WaveguideCTL
Control files for calculating the band structure for photonic crystal waveguides (originally written by [Sebastian Schulz et al](https://github.com/sschulz365/PhC_Optimization) using [MPB](https://github.com/NanoComp/mpb).
- `W1_2D_v04.ctl`: file for 2D MPB simulations
- `W1_3D_v1.ctl`: file for 3D MPB simulations (significantly slower)