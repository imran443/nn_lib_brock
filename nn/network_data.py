'''
Created on Apr 12, 2017

@author: Matth
'''
import numpy as np
from _operator import lt

class network_data():
    epochs = 1
    learningRate = 0.3
    randomRange = 0.5 #the range that neuron weight values can be randomized to (0.5 equals the range -0.5 to 0.5)
    
    learningTechnique = "backprop"
    
    momentumUse = False
    momentumAlpha = 0.2
    biasUse = False
    biasRange = 0.5 #similar to randomRange
    
    #The layers that contain connection weight values, this is initialized on networkdata creation
    #layerWeights[0] are the connections between input and first hidden layers
    layerWeights = None
    
    layerInput = None
    layerOutput = None
    
    networkLayers = None #will contain the raw layer data, this is used to determine set layer activation funtions
    
    
    #def __init__(self):
        
        
    #initializes the networks data class
    #Accepts a 2D list of layer information, each item in "layers" contains the layer size, and its activation function
    def buildNetwork(self, layers):
        networkLayers = layers
        numOfLayers = len(layers)
        
        #initialize our nparray lists
        #Summation, Derivative and other arrays do not need initialization
        
        #Create the input and output layers
        layerInput = np.zeros((1,layers[0][0]))
        layerOutput = np.zeros((1,layers[numOfLayers-1][0]))
        
        #Create the connection weights between layers
        layerWeights = [None] * numOfLayers

        
        for i in range (1, numOfLayers):
            layerWeights[i] = np.random.uniform(-self.randomRange, self.randomRange, (layers[i-1][0],layers[i][0]))
        
        #Create random bias values for each layer
        layerBias = [None] * numOfLayers
        
        for i in range (1, numOfLayers):
            layerBias[i] = np.random.uniform(-self.biasRange, self.biasRange, (1,layers[i][0]))
        
        
    
        print ("Network created with " + str(numOfLayers) + " layers")
        
    ######
    ##SETTER FUNCTIONS
    ######
    def setLearningTechnique(self, lt):
        learningTechnique = lt
        print ("Learning technique set to " + lt)
        
    def setLearningRate(self, lr):
        learningRate  = lr
        print ("Learning rate set to " + str(lr))
        
    def setEpochs(self, ep):
        epochs = ep
        print ("Number of epochs set to " + str(ep))
        
    def setRandomRange(self, rr):
        randomRange = rr
        print ("Neuron initialized weight values will be randomly set between -" + str(rr) + " and " + str(rr))
        
    def setMomentum(self, activate, alpha):
        momentumUse = activate
        momentumAlpha = alpha
        if (activate):
            print ("Using momentum with an alpha of " + str(alpha))
        else:
            print ("Not using momentum")
        
    def setBias(self, activate, amt):
        biasUse = activate
        biasRange = amt
        if (activate):
            print ("Using bias with a weight of " + str(amt))
        else:
            print ("Not using bias")
            
    def setLayers(self, layers):
        networkLayers = layers;
        
    
        