'''
Created on Apr 21, 2017

@author: Matth

The master class users will import into their projects. They will only call commands from this
class, where this class will manage the network and perform the desired actions
'''
from nn.network_data import network_data
from nn.network_trainer import network_trainer
import sys

class network_master():
    
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
    
    
    # Creates our network data object and sets its values accordingly
    def createNetwork(self, learningTechnique, layers, learningRate, randWeightRange):
        global nd
        
        nd = network_data()
        nd.setLearningTechnique(learningTechnique)
        nd.setLearningRate(learningRate)
        nd.setRandomRange(randWeightRange)
        
        
        print() # To space out our console text
        nd.buildNetwork(layers) #always build the network after setting network values
        print()
    
    
    # Creates our network trainer object, passes it our network data
    #calls its training method
    def trainNetwork(self):
        global nd
        
        if (nd.trainingData is None):
            sys.stderr.write("  ERROR: Must load training data!")
            return
            
        nt = network_trainer(nd)
        
        if (nd.holdoutTechnique == "holdout"):
            nt.holdout()
            
        elif(nd.holdoutTechnique == "kfold"):
            nt.kfold()
        

##RAW CODE FOR TESTING PURPOSES
layers = [[4,"sigmoid"],[3,"sigmoid"],[1,"sigmoid"]]

testNetwork = network_master()
testNetwork.createNetwork("backprop", layers, 0.2, 0.6)
testNetwork.loadData("parity4.txt","parity4Expected.txt", ',')
testNetwork.trainNetwork()
        
        
        