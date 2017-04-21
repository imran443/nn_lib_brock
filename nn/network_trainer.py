'''
Created on Apr 21, 2017

@author: Matth
'''

class network_trainer():
    
    def __init__(self, networkData):
        nd = networkData
        
        print ("Trainer instantiated.")
        
        
    def trainNetwork(self):
        print ("Training Network!")
        
        self.forwardPass()
        
        if (self.nd.trainingTechnique == "backprop"):
            
            self.backPropagation()
            
        elif (self.nd.trainingTechnique == 'rprop'):
            
            self.rPropagation()
            
            
    #The base forward pass of the network
    def forwardPass(self):
        break
    
    def backPropagation(self):
        break
    
    def rPropagation(self):
        break