# Anson Ho, 2021
# Python script for generating training sets for the neural network

from backend.experiment import W1Experiment
from backend import constraintsFix
from backend import objectiveFunctions
from random import uniform
import pandas as pd
import sys

# constraints fixer often takes many recursion steps to work
sys.setrecursionlimit(1000)

def main():
    # paths to files
    mpb = "/usr/bin/mpb"
    inputFile = "/home/nanophotgrp/Desktop/PCO/WaveguideCTL/W1_2D_v04.ctl.txt"
    outputLog = "/home/nanophotgrp/Desktop/PCO/test-run.out"
    outputCSV = "/home/nanophotgrp/Desktop/PCO/2021-03-10_train-set-6.csv"

    # initialise parameter map
    # pars = {'r0': 0.1, 'r1': 0.2, 'r2': 0.2845, 'r3': 0.290696, 's3': -0.004733, 's2': -0.000407, 's1': -0.048862, 'p1':0.3, 'p2':0.3, 'p3':0.3}
    initPars = {'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0, 's1': 0, 's2': 0, 's3': 0, 'p1': 0, 'p2': 0, 'p3': 0}
    constraintFunctions = [constraintsFix.latticeConstraintsLD]

    # number of runs
    runs = 10
    outputData = []

    # create datasets
    for i in range(runs):
        print("STARTING RUN {} OUT OF {}".format(i+1, runs))
        
        # randomise parameters with constraints
        # make copy of initPars
        randomPars = {param: value for param,value in initPars.items()}
        randomPars['r0'] = uniform(0.2, 0.4)
        randomPars['r1'] = uniform(0.2, 0.4)
        randomPars['r2'] = uniform(0.2, 0.4)
        randomPars['r3'] = uniform(0.2, 0.4)
        randomPars['s1'] = uniform(-0.5, 0.5)
        randomPars['s2'] = uniform(-0.5, 0.5)
        randomPars['s3'] = uniform(-0.5, 0.5)
        randomPars['p1'] = uniform(-0.5, 0.5)
        randomPars['p2'] = uniform(-0.5, 0.5)
        randomPars['p3'] = uniform(-0.5, 0.5)
        #print("BEFORE", randomPars) # debug
        
        constrainPars = constraintsFix.fix(randomPars, constraintFunctions)
        #print("AFTER", constrainPars) # debug
        
        #if constrainPars == initPars
        
        # set up experiment
        experiment = W1Experiment(mpb, inputFile, outputLog)
        experiment.setParams(constrainPars)
        experiment.setCalculationType(4) # fails with other calculation types
        experiment.setBand(23)
        experiment.kinterp = 19
        experiment.dim3 = False # 2D
        experiment.split = "-split 15"

        # run experiment and extract FOMs
        getFOMs = experiment.extractFunctionParams()
        #print(foms) # debug
        
        # combine dictionaries
        outputDict = {**constrainPars, **getFOMs}
        outputData.append(outputDict)
        print(outputDict)
        
        print("END OF RUN NO. {} \n".format(i+1))

    # print(outputData) # debug
    df = pd.DataFrame(outputData)
    df.to_csv(outputCSV)

if __name__ == "__main__":
    main()
