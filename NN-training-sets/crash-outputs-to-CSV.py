# 2021 Anson Ho
import pandas as pd
from ast import literal_eval

inputTXT = "/Users/apple/desktop/photonic-crystals-neural-networks/NN-training-sets/crash-outputs.txt"
outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/NN-training-sets/2021-03-11_crash-data.csv"

crashFile = open(inputTXT, "r")
outputData = []

def main():
    for line in crashFile:
        if line[0] == "{":
            line_to_dict = literal_eval(line)
            outputData.append(line_to_dict)
    
    df = pd.DataFrame(outputData)
    df.to_csv(outputCSV)

if __name__ == "__main__":
    main()
