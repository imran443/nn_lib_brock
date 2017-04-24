'''
Created on Apr 21, 2017

@author: Matthew Kirchhof, Imran Qureshi

The master class users will import into their projects. They will only call commands from this
class, where this class will manage the network and perform the desired actions
'''
from brocknet import network_data
from brocknet import network_trainer
import sys
import network_tester

class NetworkMaster:
    
    global nd, nt
    
    
    def __init__(self):    
        return

    def loadData(self, fileName, expectedName, delim):
        """
        Accepts two files, one with the training data and one with the expected results
        Sends the data to network_data
        """
        global nd
        
        # If nd doesn't exist (user hasn't initialized the network data), catch the error
        try:
            nd.loadTrainingData(fileName, expectedName, delim)
            print ("Data loaded...")
            #self.printLoadedData()
            
        except NameError:
            sys.stderr.write("  ERROR: Must create network before loading in data!")
            
    
    def printLoadedData(self):
        """Helper function to print to console the generated input"""
        global nd
        
        for item in nd.trainingData:
            print(item.inputData, item.expectedOutput)
    
    
    def detailedOutput(self, toPrint):
        """Turn detailed printing on or off"""
        global nd
        
        nd.setPrinting(toPrint)
    
    
    def createNetwork(self, layers, 
                      learningTechnique="backprop", 
                      holdoutTechnique="holdout",
                      epochs = 100, 
                      learningRate=0.3,
                      weightRange=0.5,
                      momentum=False, momentumAlpha=0.0, 
                      bias=False, biasRange = 0.0,
                      weightDecay=False, weightDecayFactor=0.05
                      ):
        """
        Creates our network data object and sets its values accordingly
        Learning technique and layers are required. All other settings are optional
        Defaults are preset here
        """
        global nd
        
        nd = network_data.NetworkData()
        
        nd.setLearningTechnique(learningTechnique)
        nd.setEpochs(epochs)
        nd.setLearningRate(learningRate)
        nd.setRandomRange(weightRange)
        
        nd.setMomentum(momentum, momentumAlpha)
        nd.setBias(bias, biasRange)
        nd.setWeightDecay(weightDecay, weightDecayFactor)
        
        
        print() # To space out our console text
        nd.buildNetwork(layers) # Always build the network after setting network values
        print()
        
    
    def set(self,learningTechnique=-1,layers=-1,holdoutTechnique=-1,holdoutAmt=-1,epochs=-1,learningRate=-1,weightRange=-1,momentum=-1, momentumAlpha=-1,bias=-1, biasRange=-1, weightDecay=-1, weightDecayFactor=-1):
        """Changes network settings to the passed parameters"""
        
        if (learningTechnique!=-1):
            nd.setLearningTechnique(learningTechnique)
        if (layers!=-1):
            nd.setLayers(layers)
        if (holdoutTechnique!=-1 and holdoutAmt!=-1):
            nd.setHoldoutTechnique(holdoutTechnique,holdoutAmt)
        if (epochs!=-1):
            nd.setEpochs(epochs)
        if (learningRate!=-1):
            nd.setLearningRate(learningRate)
        if (weightRange!=-1):
            nd.setRandomRange(weightRange)
        if (momentum!=-1 and momentumAlpha!=-1):
            nd.setMomentum(momentum, momentumAlpha)
        if (bias!=-1 and biasRange!=-1):
            nd.setBias(bias, biasRange)
        if (weightDecay!=-1 and weightDecayFactor!=-1):
            nd.setWeightDecay(weightDecay, weightDecayFactor)
        
    
    def trainNetwork(self):
        """Creates our network trainer object, passes it our network data and calls the specified holdout training method"""
        global nd, nt
        
        if (nd.trainingData is None):
            sys.stderr.write("  ERROR: Must load training data!")
            return
            
        nt = network_trainer.NetworkTrainer(nd)
        
        if (nd.holdoutTechnique == "holdout"):
            nt.holdout()
            
        elif(nd.holdoutTechnique == "kfold"):
            nt.kfold()
            
    def testNetwork(self, fileName, delim):
        global nd, nt
        
        try:
            print ("Testing Network..")
            
            ntest = network_tester.NetworkTester(nd, nt)
            ntest.testNetwork(fileName, delim)
            
        except NameError:
            sys.stderr.write("  ERROR: Must create/train network before testing!")
        

##RAW CODE FOR TESTING PURPOSES
 
layers = [[4,"sigmoid"],[3,"sigmoid"],[1,"sigmoid"]]
 
testNetwork = NetworkMaster()
 
testNetwork.createNetwork(layers, learningRate=0.2, weightRange=0.6)
 
testNetwork.loadData("parity4.txt","parity4Expected.txt", ',')
    
testNetwork.trainNetwork()

testNetwork.testNetwork("parity4.txt", ",")
        
        
        