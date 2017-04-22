'''
Created on Apr 21, 2017

@author: Matth
'''
import numpy as np

class network_trainer():
    global nd
    
    def __init__(self, networkData):
        global nd
        nd = networkData
        
        print ("Trainer instantiated.")
        
        
    def trainNetwork(self):
        global nd
        print ("Training Network!")
        
        
        for item in range(0,nd.trainingData.shape[0],2):

            self.loadTrainingData(nd.trainingData[item]) #load in our input
            self.loadExpectedOutput(nd.trainingDataExpected[item]) #set our expected output array
            
            self.forwardPass()
            
            if (nd.trainingTechnique == "backprop"):
                
                self.backPropagation()
                
            elif (nd.trainingTechnique == 'rprop'):
                
                self.rPropagation()
            
            
    #The base forward pass of the network
    def forwardPass(self):
        global nd
        
        #runs for every layer excluding input layer
        #first calculates the current layers sums, then calculates the activated values (passed through sigmoid or tanh) including bias values
        for i in range(1, nd.numOfLayers):
            nd.layerSums[i] = np.dot(nd.layerActivations[i-1], nd.layerWeights[i-1])
            nd.layerActivations[i] = self.activationFunction(nd.layerSums[i] + nd.layerBias[i], nd.networkLayers[i][1], False) 
        
    
    # Calculates the error in each output node
    def calcErrorAtOutput(self):
        global nd
        
        # Create the error contribution array for the output layer
        nd.layerError = np.empty([1, nd.layerActivations[nd.numOfLayers-1].shape[1]])

        for i in range(nd.layerActivations[nd.numOfLayers-1].shape[1]):
            output = nd.layerActivations[nd.numOfLayers-1][0, i]
            target = nd.layerOutputTarget[0, i]

            # Loads errors in to an array
            nd.layerError[0,i] = output - target
            
    
    def backPropagation(self):
        self.calcErrorAtOutput()
    
    
    def rPropagation(self):
        return
    
    
    def loadTrainingData(self, input):
        print (input)
    
    ##########
    ##neuron specific methods
    ##########
    
    #All activation functions along with their derived counterparts
    def activationFunction(self, x, func ,derive = False):

        if(func == "sigmoid"):
            if(derive == True):

                return x*(1-x)

            return 1/(1+np.exp(-x))

        elif (func == "tanh"):
            if(derive == True):

                return 1 - (np.tanh(x)) ** 2

            return np.tanh(x)