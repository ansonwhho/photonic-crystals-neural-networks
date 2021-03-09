# Design of photonic crystal waveguides using neural networks
Repository for my undergraduate dissertation, *Design of photonic crystal waveguides using neural networks*. The aim is to optimise figures of merit (FOMs, e.g. group index, propagation loss) of photonic crystal waveguides (PhCWs) while guiding slow light using neural networks. 

## Contents
This repo contains: 
- Python and Scheme files for calculating the band structure for PhCWs (originally written by Sean Billings, Sebastian Schulz et al - the code can be found [here](https://github.com/sschulz365/PhC_Optimization)) using [MIT Photonic Bands](https://github.com/NanoComp/mpb) (MPB) software. I made some edits to them so that they would be compatible with Python 3 and to obtain the desired FOMs from the MPB calculations.
- Python scripts to: 
  - Generate training data for the neural network by running MPB on randomised PhCW designs
  - Train a neural network on the data
- Others  

*This is still a work in progress.*
