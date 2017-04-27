import sys
import numpy as np
from brocknet import training_example

class NetworkTester:
    nd = None
    nt = None
    testingData = None
    
    def __init__(self, networkData, networkTrainer):
        self.nd = networkData
        self.nt = networkTrainer
        
        
    def testNetwork(self, fname, delim):
        """
        Loads training data from the textfile 'fname' by line, split via a specified delimiter
        Both lists must be in the same order, where trainingData[i] should match the trainingDataExpected[i] data
        """
        
        #Create our testing set of data
        try:
            self.testingData = np.loadtxt(fname, delimiter=delim)
            
        except:
            sys.stderr.write("ERROR: Issue loading testing data, please double check file name and delimiter")
            
        
        #Test on the dataset
        
        self.runData()
        
    
    def runData(self):

        for i in range (len(self.testingData)):            
            self.nt.setInputs(np.array(self.testingData[i])) 
                
            self.nt.forwardPass()
            
            print ("")
            print ("Given input: " + str(self.testingData[i]))
            print ("Network output results: " + str(self.nd.layerActivations[self.nd.numOfLayers-1]))
            
        

