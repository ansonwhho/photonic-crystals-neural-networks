"""
Anson Ho, 2021

Generates training experiments to be run on
MPB given a set of candidate designs. 

"""

from backend.experiment import W1Experiment
from backend import constraintsFix
# from backend import objectiveFunctions
import random
import pandas as pd
import sys
import csv

# constraints fixer often takes many recursion steps to work
sys.setrecursionlimit(10**4)

def main():
    # paths to files
    mpb = "/usr/bin/mpb"
    ctlFile = "/home/nanophotgrp/Desktop/PCO/WaveguideCTL/W1_2D_v04.ctl.txt"
    outputLog = "/home/nanophotgrp/Desktop/PCO/test-run.out"
    inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/designs/candidates/2021-04-11_candidate-set-1.csv"
    outputCSV = "/home/nanophotgrp/Desktop/PCO/2021-04-11_candidate-set-1-TEST.csv"

    constraintFunctions = [constraintsFix.latticeConstraintsLD]

    # Read in CSV file of candidate PhCW designs
    candidate_df = pd.read_csv(inputCSV, index_col=0)
    candidates = candidate_df.to_dict('records')
    runs = len(candidates)
    outputData = []

    for candidate in candidates:
        print("STARTING RUN {} OUT OF {}".format(i+1, runs))
        
        # Apply constraints
        constrainPars = constraintsFix.fix(candidate, constraintFunctions)
        
        # Set up experiment
        experiment = W1Experiment(mpb, ctlFile, outputLog)
        experiment.setParams(constrainPars)
        experiment.setCalculationType(4) # fails with other calculation types
        experiment.setBand(23)
        experiment.kinterp = 19

        experiment.dim3 = False # 2D
        experiment.split = "-split 15"

        # Run experiment and extract FOMs
        getFOMs = experiment.extractFunctionParams()
        
        # Combine dictionaries
        outputDict = {**constrainPars, **getFOMs}
        outputData.append(outputDict)
        print(outputDict)
        
        print("END OF RUN NO. {} \n".format(i+1))

    df = pd.DataFrame(outputData)
    df.to_csv(outputCSV)

if __name__ == "__main__":
    main()
