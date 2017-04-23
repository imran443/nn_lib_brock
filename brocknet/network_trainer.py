
import numpy as np
import sys

class NetworkTrainer():
    global nd
    
    def __init__(self, networkData):
        global nd
        nd = networkData
        
        print ("Trainer instantiated.")
    
    
    def holdout(self):
        global nd
        
        for i in range (1):
            
            np.random.shuffle(nd.trainingData)
            
            splitPt = int( len(nd.trainingData) * (1-nd.holdoutPercent) )
            
            trainingSet = nd.trainingData[0:splitPt]
            testingSet = nd.trainingData[splitPt+1:len(nd.trainingData)]
            
            self.trainNetwork(trainingSet)
    
    
    def kfold(self):
        global nd
        
        for i in range (nd.epochs):
            
            return #INCOMPLETE
        
    # This function is used to train the network.
    def trainNetwork(self, trainingSet):
        global nd
        # Set the expected output array for error calculations later on.
        nd.layerOutputTarget
        
        for i in range(len(trainingSet)):
            # Load in our input
            self.setInputs(trainingSet[i][0])
            
            # Set our expected output array
            self.setExpectedOutput(trainingSet[i][1]) 
            
            self.forwardPass()
            
            if (nd.trainingTechnique == "backprop"):
                
                self.backPropagation()
                
            elif (nd.trainingTechnique == 'rprop'):
                
                self.rPropagation()
            
     
    # The base forward pass of the network
    def forwardPass(self):
        global nd
        
        # Runs for n hidden layers and the 1 output layer.
        # First calculates the current layers sums, then calculates the activated values (passed through sigmoid or tanh) including bias values
        for i in range(1, nd.numOfLayers):
            nd.layerSums[i-1] = np.dot(nd.layerActivations[i-1], nd.layerWeights[i-1])
            nd.layerActivations[i] = self.activationFunction(nd.layerSums[i-1] + nd.layerBias[i], nd.networkLayers[i][1], False)
        
    
    # Calculates the error in each output node
    def calcErrorAtOutput(self):
        global nd
        outputLayer = nd.layerActivations[nd.numOfLayers-1]
        
        # Create the error contribution array for the output layer
        nd.layerError = np.empty([1, outputLayer.shape[1]])
        
        for i in range(outputLayer.shape[1]):
            output = outputLayer[0, i]
            target = nd.layerOutputTarget[0, i]

            # Loads errors in to an array
            nd.layerError[0,i] = output - target
            
        
    # Simple back propagation 
    def backPropagation(self):
        self.calcErrorAtOutput()
    
    
    def rPropagation(self):
        return
    
    # Sets the input vector to the as the input layer.
    def setInputs(self, inputVector):
        global nd
        
        try:
            nd.layerActivations[0] = inputVector
        except:
            sys.stderr.write("  ERROR: Mismatched input and output. Please ensure your input nodes match the size of your input data!")
        
        
    def setExpectedOutput(self, expectedVector):
        global nd
        
        try:
            nd.layerOutputTarget = expectedVector
        except:
            sys.stderr.write("  ERROR: Mismatched input and output. Please ensure your input nodes match the size of your input data!")
    

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