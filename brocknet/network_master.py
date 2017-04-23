'''
Created on Apr 21, 2017

@author: Matth

The master class users will import into their projects. They will only call commands from this
class, where this class will manage the network and perform the desired actions
'''
from network_data import NetworkData
from network_trainer import NetworkTrainer
import sys

class NetworkMaster():
    
    global nd, nt
    
    
    def __init__(self):
        print("test")       
        return

    # Accepts two files, one with the training data and one with the expected results
    # Sends the data to network_data
    def loadData(self, fileName, expectedName, delim):
        global nd
        
        # If nd doesn't exist (user hasn't initialized the network data), catch the error
        try:
            nd.loadTrainingData(fileName, expectedName, delim)
            print ("Data loaded...")
            self.printLoadedData()
            
        except NameError:
            sys.stderr.write("  ERROR: Must create network before loading in data!")
            return
    
    # Helper function to print to console the generated input
    def printLoadedData(self):
        global nd
        
        for item in nd.trainingData:
            print(item[0], item[1])
    
    
    #Creates our network data object and sets its values accordingly
    #Learning technique and layers are required. All other settings are optional
    def createNetwork(self, learningTechnique, layers, 
                      holdoutTechnique="holdout",
                      epochs = 50, 
                      learningRate=0.3, 
                      weightRange=0.5, 
                      momentum=False, momentumAlpha=0.0, 
                      bias=False, biasRange = 0.0
                      ):
        global nd
        
        nd = NetworkData()
        
        nd.setLearningTechnique(learningTechnique)
        nd.setEpochs(epochs)
        nd.setLearningRate(learningRate)
        nd.setRandomRange(weightRange)
        
        nd.setMomentum(momentum, momentumAlpha)
        nd.setBias(bias, biasRange)
        
        
        print() # To space out our console text
        nd.buildNetwork(layers) #always build the network after setting network values
        print()
        
    
    def set(self,learningTechnique=-1,layers=-1,holdoutTechnique=-1,holdoutAmt=-1,epochs=-1,learningRate=-1,weightRange=-1,momentum=-1, momentumAlpha=-1,bias=-1, biasRange=-1):
        
        if (learningTechnique!=-1):
            nd.setLearningTechnique = learningTechnique
        if (layers!=-1):
            nd.setLayers(layers)
        if (holdoutTechnique!=-1 and holdoutAmt!=-1):
            nd.setHoldoutTechnique(holdoutTechnique,holdoutAmt)
        if (epochs!=-1):
            nd.setEpochs = epochs
        if (learningRate!=-1):
            nd.setLearningRate = learningRate
        if (weightRange!=-1):
            nd.setRandomRange = weightRange
        if (momentum!=-1 and momentumAlpha!=-1):
            nd.setMomentum(momentum, momentumAlpha)
        if (bias!=-1 and biasRange!=-1):
            nd.setBias(bias, biasRange)
        
    
    
    # Creates our network trainer object, passes it our network data
    #calls its training method
    def trainNetwork(self):
        global nd
        
        if (nd.trainingData is None):
            sys.stderr.write("  ERROR: Must load training data!")
            return
            
        nt = NetworkTrainer(nd)
        
        if (nd.holdoutTechnique == "holdout"):
            nt.holdout()
            
        elif(nd.holdoutTechnique == "kfold"):
            nt.kfold()
        

##RAW CODE FOR TESTING PURPOSES

layers = [[4,"sigmoid"],[3,"sigmoid"],[1,"sigmoid"]]

testNetwork = NetworkMaster()

testNetwork.createNetwork("backprop", layers, learningRate=0.2, weightRange=0.6)

testNetwork.loadData("parity4.txt","parity4Expected.txt", ',')

testNetwork.set(momentum=True, momentumAlpha=0.3)

testNetwork.trainNetwork()
        
        
        