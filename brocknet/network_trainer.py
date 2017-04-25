'''
Created on Apr 21, 2017

@author: Matthew Kirchhof, Imran Qureshi

The class which trains a NetworkData object with its selected settings
'''
import numpy as np
import sys

class NetworkTrainer:
    global nd
    
    accuracyCount = 0
    np.random.seed(1)
    
    def __init__(self, networkData):
        global nd
        nd = networkData
    
    def holdout(self):
        """
        Withholds holdoutAmt percent of the data from training for use in testing
        """
        global nd
        # Randomize the training data before split.
        np.random.shuffle(nd.trainingData)
        
        splitPt = int( len(nd.trainingData) * (1-nd.holdoutAmt) )
        
        trainingSet = nd.trainingData[0:splitPt]
        testingSet = nd.trainingData[splitPt+1:len(nd.trainingData)]
        
        for i in range (1):
            print("Epoch " + str(i+1) +": \n")
            
            np.random.shuffle(trainingSet)
            
            # Trains the network for all data sets 
            self.trainNetwork(trainingSet)
    
    
    def kfold(self):
        """
        Runs epoch number of times
        shuffles the data for each epoch, splits the data into holdoutAmt lists and runs system training on all subsets besides the current k set.
        """
        
        global nd
        
        # For every epoch
        for i in range (nd.epochs):
            
            np.random.shuffle(nd.trainingData)
            
            splitTrainingSets = np.array_split(nd.trainingData,nd.holdoutAmt)
            
            #run through training kfold times, each withholding the jth subset for testing
            for j in range (nd.holdoutAmt):
                
                trainingSet = []
                
                #split into this runs training and testing sets
                testingSet = splitTrainingSets[j]
                
                #combine remaining items into training set
                for a,item in enumerate(splitTrainingSets):
                    if (a != j):
                        
                        trainingSet.extend(item)  
                #train
                self.trainNetwork(trainingSet)
            
        
    # This function is used to train the network.
    def trainNetwork(self, trainingSet):
        """ 
        Controls the training algorithms used and overall structure of the networks training
        Accepts a list of TrainingExample objects to train the network on
        """
        
        global nd
        
        for i in range(1):
            # Load in our input
            self.setInputs(trainingSet[i].inputData) 
            
            # Special case of one output node, must make a single element list
            if (nd.networkLayers[nd.numOfLayers-1][0] == 1):
                self.setExpectedOutput(np.array([trainingSet[i].expectedOutput]))
            else:
                self.setExpectedOutput(np.array(trainingSet[i].expectedOutput))
            
            self.forwardPass()
            self.accuracyOfNN()
            
            print("Layer Activations: \n", nd.layerActivations)
            print("Layer Weights: \n", nd.layerWeights)
             
            if (nd.trainingTechnique == "backprop"):
                self.backPropagation()
                self.updateWeights()
                 
            elif (nd.trainingTechnique == 'rprop'):
                self.rPropagation()
                
                
    def forwardPass(self):
        """
        The base forward pass of the network
        """
        global nd
        
        # Runs for n hidden layers and the 1 output layer.
        # First calculates the current layers sums, then calculates the activated values (passed through sigmoid or tanh) including bias values
        for i in range(1, nd.numOfLayers):
            nd.layerSums[i-1] = np.dot(nd.layerActivations[i-1], nd.layerWeights[i-1])
            nd.layerActivations[i] = self.activationFunction(nd.layerSums[i-1] + nd.layerBias[i], nd.networkLayers[i][1], False)
        
    
    
    def calcErrorAtOutput(self):
        """ 
        Calculates the error in each output node
        """
        global nd
        
        # Grab the output layer
        outputLayer = nd.layerActivations[nd.numOfLayers-1]
        
        # Create the error contribution array for the output layer
        errContribution = np.empty([1, outputLayer.shape[1]])

        for i in range(outputLayer.shape[1]):
            try:
                output = outputLayer[0, i]
                target = nd.layerOutputTarget[0, i]
                # Stores the error for each output node into a array
                errContribution[0, i] = output - target
            except:
                sys.stderr.write("  ERROR: the expected output file and output layer do not match!")
        
        # Stores the errContribution in a list.
        nd.layerErrors[nd.numOfLayers-2] = errContribution  
        
    def backPropagation(self):
        """ 
        Simple back propagation method to help calculate the gradients.
        """
        
        # Gets the error contribution at the output layer only.
        self.calcErrorAtOutput()
        
        # The error contribution at the output layer.
        errContribution = nd.layerErrors[nd.numOfLayers-2]
        
        # Grab the output layer, which has the squashed values.
        outputLayer = nd.layerActivations[nd.numOfLayers-1]
        
        print("Output Layer: \n", outputLayer)
        print("Expected Layer; \n", nd.layerOutputTarget)
        # Send output values into derivative of activation function.
        deriveOutput = self.activationFunction(outputLayer, "sigmoid", True)
        
        print("Derived Output Layer: \n", deriveOutput)
        print("Error Output Layer: \n", errContribution)
        
        # Multiply the derived values with the error for the output layer.
        derivAndErr = deriveOutput * errContribution
        
        print("Derivative and Error: \n", derivAndErr)
        
        # Transpose the array for easier matrix multiplication.
        transDerivAndErr = np.matrix.transpose(derivAndErr)
        
        # Multiply with hidden layer which is before the output layer in the structure.
        deltaWeightsHtoO = np.dot(transDerivAndErr, nd.layerActivations[nd.numOfLayers-2])
        
        # Transpose again to fix the alignment of values.
        nd.layerGradients[nd.numOfLayers-2] = np.matrix.transpose(deltaWeightsHtoO)
        
        print("Delta weights HtoO: \n ", nd.layerGradients[nd.numOfLayers-2])
        
        # Start at 2, the layerWeights is 1 size less than the layerActivation list.
        for i in range(2, nd.numOfLayers):
            print("Hidden Layer " + str(i-1) + ": \n")
            # The second counter used to store the error value.
            j = 3
            
            # Transpose hidden to output weight matrix.
            hiddenWeightTrans = np.matrix.transpose(nd.layerWeights[nd.numOfLayers-i])
            
            # Calculates the error at the hidden layer using the weight connections of this layer and the next.
            errContributionHidden = np.dot(nd.layerErrors[nd.numOfLayers-i], hiddenWeightTrans)
            
            print("errContributionHidden: \n", errContributionHidden)
            
            # Stores the hidden layer error.
            nd.layerErrors[nd.numOfLayers-j] =  errContributionHidden
            
            # Derive the hidden layer values.
            derivHidden = self.activationFunction(nd.layerActivations[nd.numOfLayers-i], "sigmoid", True)
            
            # Multiply the derivative of the hidden multiplied with the error of that hidden layer.
            hiddenDerivAndErr = derivHidden * errContributionHidden
            
            print("hiddenDerivAndErr: \n", hiddenDerivAndErr)
            
            transHiddenDerivAndErr = np.matrix.transpose(hiddenDerivAndErr)
            
            deltaWeightsItoH = np.dot(transHiddenDerivAndErr, nd.layerActivations[nd.numOfLayers-j])
            
            deltaWeightsItoH = np.matrix.transpose(deltaWeightsItoH)
            
            print("deltaWeightsItoH: \n", deltaWeightsItoH)
            
            # Store the hidden layers gradients. 
            nd.layerGradients[nd.numOfLayers-j] = hiddenDerivAndErr
            
            j+=1
        
    def updateWeights(self):
        """ Used to update the weights of each layer after a certain learning technique."""
        
        # For each set of weights per layer, update them.
        for i in range(len(nd.layerWeights)):
            nd.layerWeights[i] = nd.layerWeights[i] - nd.learningRate * nd.layerGradients[i]
        
        print("Updated Weights: \n", nd.layerWeights)
        
    
    def rPropagation(self):
        return
    

    def decayWeights(self):
        """Decays every connections weight by weightDecayFactor percent"""
        
        for l in range (nd.numOfLayers-2):
            nd.layerWeights[l] = nd.layerWeights[l]*(1-nd.weightDecayFactor)
    
    
    def setInputs(self, inputVector):
        """Sets the first layers activation layer to the current input training example"""
        
        global nd
        
        # Resize the inputVector to have a (row, columns) for later dot product operations.
        inputVector = np.resize(inputVector,(1, inputVector.shape[0]))
        
        try:
            nd.layerActivations[0] = inputVector

        except:
            sys.stderr.write("  ERROR: Mismatched input and output. Please ensure your input nodes match the size of your input data!")
        
        
    def setExpectedOutput(self, expectedValue):
        """Sets the last layers expected outputs to the current input training examples expected output"""
        global nd
        
        # Resize the inputVector to have a (row, columns) for later dot product operations.
        expectedValue = np.resize(expectedValue,(1, expectedValue.shape[0]))
        
        try:
            nd.layerOutputTarget = expectedValue
        except:
            sys.stderr.write("  ERROR: Mismatched input and output. Please ensure your input nodes match the size of your input data!")
    
    def accuracyOfNN(self):
        """ Used to check the accuracy of the network by how many samples it correctly identifies."""
        outputLayer = nd.layerActivations[nd.numOfLayers-1].copy()
        
        # Step wise function for the output layer
        for i in range(outputLayer.shape[1]):
            if(outputLayer[0, i] > 0.5):
                outputLayer[0, i] = 1
            elif(outputLayer[0, i]<0.5):
                outputLayer[0, i] = 0
        
        # Compare the current output to the expected to check accuracy.
        if(np.array_equal(outputLayer, nd.layerOutputTarget)):
            self.accuracyCount+=1
            print("Accuracy Count: ", self.accuracyCount)
        
    ##########
    ##neuron specific methods
    ##########
    
    def activationFunction(self, x, func ,derive = False):
        """All activation functions along with their derived counterparts"""
        
        if(func == "sigmoid"):
            if(derive == True):

                return x*(1-x)

            return 1/(1+np.exp(-x))

        elif (func == "tanh"):
            if(derive == True):

                return 1 - (np.tanh(x)) ** 2

            return np.tanh(x)