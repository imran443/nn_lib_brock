'''
Created on Apr 12, 2017

@author: Matth
'''
from test.test_math import eps

class network_data():
    epochs = 1
    learningRate = 0.3
    randomRange = 0.5 #the range that neuron weight values can be randomized to (0.5 equals the range -0.5 to 0.5)
    
    momentumUse = False
    momentumAlpha = 0.2
    biasUse = False
    biasAmount = 1
    
    networkLayers = []
    
    
    def __init__(self):
        print ("Network Initialized...")
        
            
        
    def setLearningRate(self, lr):
        learningRate  = lr
        print ("Learning rate set to " + lr)
        
    def setEpochs(self, ep):
        epochs = ep
        print ("Number of epochs set to " + ep)
        
    def setRandomRange(self, rr):
        randomRange = rr
        print ("Neuron initialized weight values will be randomly set between -" + rr + " and " + rr)
        
    def setMomentum(self, activate, alpha):
        momentumUse = activate
        momentumAlpha = alpha
        if (activate):
            print ("Using momentum with an alpha of " + alpha)
        else:
            print ("Not using momentum")
        
    def setBias(self, activate, amt):
        biasUse = activate
        biasAmount = amt
        if (activate):
            print ("Using bias with a weight of " + amt)
        else:
            print ("Not using bias")
            
    def setLayers(self, layers):
        networkLayers = layers;
        
    
        