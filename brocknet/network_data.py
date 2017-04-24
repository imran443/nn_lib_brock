'''
Created on Apr 21, 2017

@author: Matthew Kirchhof, Imran Qureshi

The main object which holds all network information and settings
'''

import numpy as np
import sys
from brocknet import training_example



class NetworkData:
    printInfo = True

    numOfLayers = 0

    epochs = 1
    learningRate = 0.3
    randomRange = 0.5 # The range that neuron weight values can be randomized to (0.5 equals the range -0.5 to 0.5)
    
    trainingTechnique = "backprop"
    holdoutTechnique = "holdout"
    holdoutAmt = 0.3 # The percentage OR number of k folds, dependent on the hold out technique above
    
    momentumUse = False
    momentumAlpha = 0.0
    biasUse = False
    biasRange = 0.0 # Similar to randomRange
    weightDecay = False
    weightDecayFactor = 0.05 #The percentage a weight decays by
    
    trainingData = None
    
    networkLayers = None # Will contain the raw layer data, this is used to determine set layer activation functions

    # The layers that contain connection weight values, this is initialized on network data creation
    # layerWeights[0] are the connections between input and first hidden layers
    layerWeights = None
    layerOutputTarget = None
    layerSums = None
    layerActivations = None
    layerErrors = None        
    layerGradients = None 
       
    def buildNetwork(self, layers):
        """
        Initializes the networks data class
        Accepts a 2D list of layer information, each item in "layers" contains the layer size, and its activation function
        """
        
        self.networkLayers = layers
        self.numOfLayers = len(layers)
        
        # Initialize our np array lists
        # Stores the sums of each layers resulting dot product.
        self.layerSums = [None] * (self.numOfLayers-1)
        
        # Stores the activation result of a nodes sums (Basically the output of nodes)
        self.layerActivations = [None] * self.numOfLayers 
        
        # Stores the connecting weights between layers.
        self.layerWeights = [None] * (self.numOfLayers - 1)
        
        # Stores the gradient values for n hidden layers and output layers.
        self.layerGradients = [None] * (self.numOfLayers - 1) 
        
        # Will hold all of the error arrays for each layer.
        self.layerErrors = [None] * (self.numOfLayers-1)
        
        # Random weights will be generated for each layers' connections
        for i in range (0, self.numOfLayers-1):
            self.layerWeights[i] = np.random.uniform(-self.randomRange, self.randomRange, (layers[i][0],layers[i+1][0]))

        # Create random bias values for each layer
        self.layerBias = [None] * self.numOfLayers
        
        for i in range (1, self.numOfLayers):
            self.layerBias[i] = np.random.uniform(-self.biasRange, self.biasRange, (1,layers[i][0]))
        
        print ("Network created with " + str(self.numOfLayers) + " layers")
    
    # Loads the data of the two specified files (Training, Expected)
    def loadTrainingData(self, fname, ename, delim):
        """
        Loads training data from the textfile 'fname' by line, split via a specified delimiter
        Also loads the training data's expected output list
        Both lists must be in the same order, where trainingData[i] should match the trainingDataExpected[i] data
        """
        
        try:
            self.trainingData = np.loadtxt(fname, delimiter=delim)
            trainingDataExpected = np.loadtxt(ename, delimiter=delim)
            
            # Combine the data together into 1 array.
            self.combineData(trainingDataExpected)
        except:
            sys.stderr.write("ERROR: Issue loading training data, please double check file name and delimiter")
            
    # Combine the data so that the expected answer is paired up with its respective data.
    def combineData(self, trainingDataExpected):
        """
        Combines each training example piece with its matching expected output into an object
        Returns the resulting list of TrainingExample objects
        """
        
        combinedTrainingData = []
        

        for i in range (len(self.trainingData)):
            temp = training_example.TrainingExample(self.trainingData[i], trainingDataExpected[i])
            combinedTrainingData.append(temp)
        self.trainingData = combinedTrainingData
        
    ######
    ##SETTER FUNCTIONS
    ######
    def setLearningTechnique(self, lt):
        print (lt)
        self.learningTechnique = lt
        print ("Learning technique set to " + lt)
        
    def setHoldoutTechnique(self, ht, val):
        self.holdoutTechnique = ht
        self.holdoutAmt = val
        
        print ("Holdout technique set to " + ht)
        
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
            
    def setWeightDecay(self, wd, wdf):
        self.weightDecay = wd
        self.weightDecayFactor = wdf
        if (wd):
            print ("Using weight decay with a weight decay percentage of " + str(wdf))
        else:
            print ("Not using weight decay")
            
    def setLayers(self, layers):
        self.networkLayers = layers;
        
    def setPrinting(self, toPrint):
        if type(toPrint == bool):
            self.printInfo = toPrint
            print ("Detailed system printing set to " + str(toPrint))
        else:
            print ("System printing NOT updated: Must pass a boolean True or False")
        
    
        