'''
Created on Apr 12, 2017

@author: Matth
'''
import numpy as np
import sys

class network_data():
    epochs = 1
    learningRate = 0.3
    randomRange = 0.5 #the range that neuron weight values can be randomized to (0.5 equals the range -0.5 to 0.5)
    
    trainingTechnique = "backprop"
    
    momentumUse = False
    momentumAlpha = 0.0
    biasUse = False
    biasRange = 0.0 #similar to randomRange
    numOfLayers = 0
    
    trainingData = None
    
    networkLayers = None #will contain the raw layer data, this is used to determine set layer activation funtions

    #The layers that contain connection weight values, this is initialized on networkdata creation
    #layerWeights[0] are the connections between input and first hidden layers
    layerWeights = None
    layerOutputTarget = None
    layerSums = None
    layerActivations = None
    layerError = None        
        
    #initializes the networks data class
    #Accepts a 2D list of layer information, each item in "layers" contains the layer size, and its activation function
    def buildNetwork(self, layers):
        self.networkLayers = layers
        self.numOfLayers = len(layers)
        
        #initialize our nparray lists
        self.layerSums = [None] * self.numOfLayers #stores the sums of a nodes inputs
        self.layerActivations = [None] * self.numOfLayers #stores the activation result of a nodes sums (Basically the output of nodes)
        self.layerWeights = [None] * self.numOfLayers #stores the connecting weights between layers
        self.layerError = [None] * self.numOfLayers
    
        
        self.layerActivations[0] = np.zeros((1,layers[0][0])) #this is our input layer, layers[0][0] is the size of the first layer the user defined
        self.layerOutputTarget = np.zeros((1,layers[self.numOfLayers-1][0])) #these values will be set according to our desired output
        
        for i in range (0, self.numOfLayers-1):
            self.layerWeights[i] = np.random.uniform(-self.randomRange, self.randomRange, (layers[i][0],layers[i+1][0]))


        
        #Create random bias values for each layer
        self.layerBias = [None] * self.numOfLayers
        
        for i in range (1, self.numOfLayers):
            self.layerBias[i] = np.random.uniform(-self.biasRange, self.biasRange, (1,layers[i][0]))
        
        
    
        print ("Network created with " + str(self.numOfLayers) + " layers")
        
    def loadTrainingData(self, fname, ename, delim):
        try:
            self.trainingData = np.loadtxt(fname, delimiter=delim)
            trainingDataExpected = np.loadtxt(ename, delimiter=delim)
            
            self.combineData(trainingDataExpected)
            
        except:
            sys.stderr.write("  ERROR: Issue loading training data, please double check file name and delimiter")
            
    
    def combineData(self, trainingDataExpected):
        combinedTrainingData = []
        
        for i in range (0, len(self.trainingData)):
            
            combinedTrainingData.append([self.trainingData[i], trainingDataExpected[i]])
                    
        self.trainingData = combinedTrainingData
    
    ######
    ##SETTER FUNCTIONS
    ######
    def setLearningTechnique(self, lt):
        self.learningTechnique = lt
        print ("Learning technique set to " + lt)
        
    def setLearningRate(self, lr):
        self.learningRate  = lr
        print ("Learning rate set to " + str(lr))
        
    def setEpochs(self, ep):
        self.epochs = ep
        print ("Number of epochs set to " + str(ep))
        
    def setRandomRange(self, rr):
        self.randomRange = rr
        print ("Neuron initialized weight values will be randomly set between -" + str(rr) + " and " + str(rr))
        
    def setMomentum(self, activate, alpha):
        self.momentumUse = activate
        self.momentumAlpha = alpha
        if (activate):
            print ("Using momentum with an alpha of " + str(alpha))
        else:
            print ("Not using momentum")
        
    def setBias(self, activate, amt):
        self.biasUse = activate
        self.biasRange = amt
        if (activate):
            print ("Using bias with a weight of " + str(amt))
        else:
            print ("Not using bias")
            
    def setLayers(self, layers):
        self.networkLayers = layers;
        
    
        