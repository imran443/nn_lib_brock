'''
Created on Apr 23, 2017

@author: Matthew Kirchhof, Imran Qureshi

Before using this library test program, you MUST install the library following the pip install example 
given in the system documents!!
'''
import brocknet

# Two nodes for output first node is 0 and the second one is 1.
myLayers = [[4,"sigmoid"],[3,"sigmoid"],[2,"sigmoid"]]

myNetwork = brocknet.network()

myNetwork.createNetwork(myLayers)

myNetwork.set(learningTechnique = "backprop",
              holdoutTechnique = "holdout",
              holdoutAmt = 0.2, 
              epochs = 2000, 
              learningRate = 0.3, 
              weightRange = 0.5, 
              momentum = False, 
              momentumAlpha = 0.2, 
              bias = False, 
              biasRange = 0.5, 
              weightDecay = False, 
              weightDecayFactor = 0.05)

myNetwork.loadData("myInputData.txt", "myInputDataExpected.txt", ",")
myNetwork.printLoadedData()

myNetwork.trainNetwork()

myNetwork.testNetwork("myInputData.txt", ",")


