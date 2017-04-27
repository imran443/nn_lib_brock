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
    #np.random.seed(1)
    
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
        
        for i in range (nd.epochs):
            print("Epoch " + str(i+1) +":")
            
            np.random.shuffle(trainingSet)
            
            # Trains the network for all data sets 
            self.trainNetwork(trainingSet)
            
            accuracy = self.accuracyOfEpoch()/len(trainingSet)
            
            print("Epoch Accuracy: \n", accuracy)
            
            # Stops when the network reaches a accuracy of 95%
            if(accuracy > nd.threshold):
                print("Best accuracy for Epoch: \n", accuracy)
                self.testNetwork(testingSet)
                accuracy = self.accuracyOfEpoch()/len(testingSet)
                print("Accuracy For Testing Set: \n", accuracy)
                print("TRAINING COMPLETE!!!")
                return
            if((i+1) == nd.epochs):
                print("Run network again, accuracy of " + str(nd.threshold) + " was not reached!")
    
    def kfold(self):
        """
        Runs epoch number of times
        shuffles the data for each epoch, splits the data into holdoutAmt lists and runs system training on all subsets besides the current k set.
        """
        
        global nd
        sumOfKRuns = 0
        # For every epoch
        for i in range (nd.epochs):
            
            np.random.shuffle(nd.trainingData)
            
            splitTrainingSets = np.array_split(nd.trainingData,nd.holdoutAmt)
            # Reset the sum
            sumOfKRuns = 0
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
                
                sumOfKRuns += self.accuracyOfEpoch()/len(trainingSet)
            
            accuracy = sumOfKRuns/nd.holdoutAmt
            print("Epoch Accuracy of k-fold: \n", accuracy)
                
            # Stops when the network reaches a accuracy of 95%
            if(accuracy > nd.threshold):
                print("Best accuracy for Epoch: \n", accuracy)
                self.testNetwork(testingSet)
                accuracy = self.accuracyOfEpoch()/len(testingSet)
                print("Accuracy For Testing Set: \n", accuracy)
                return
    
    def trainNetwork(self, trainingSet):
        """ 
        Controls the training algorithms used and overall structure of the networks training
        Accepts a list of TrainingExample objects to train the network on
        """
        
        global nd
        
        for i in range(len(trainingSet)):
            # Load in our input
            self.setInputs(trainingSet[i].inputData) 
            
            # Special case of one output node, must make a single element list
            if (nd.networkLayers[nd.numOfLayers-1][0] == 1):
                self.setExpectedOutput(np.array([trainingSet[i].expectedOutput]))
            else:
                self.setExpectedOutput(np.array(trainingSet[i].expectedOutput))
            
            self.forwardPass()
            # Check the accuracy after each forward pass.
            self.accuracyOfNN()
             
            if (nd.trainingTechnique == "backprop"):
                self.backPropagation()
                self.updateWeights()
                
            elif (nd.trainingTechnique == 'rprop' or nd.trainingTechnique == 'delta'):
                self.backPropagation()
        
        if(nd.trainingTechnique == 'rprop'):
            self.rPropagation()
            self.setSumOfPrevGradients()
            
        if(nd.trainingTechnique == 'delta'):
            self.deltaBarDelta()
            self.setSumOfPrevGradients()
    
    def testNetwork(self, testSet):          
        """ Tests the network after training. """
        
        for i in range(len(testSet)):
            # Load in our input
            self.setInputs(testSet[i].inputData) 
            
            # Special case of one output node, must make a single element list
            if (nd.networkLayers[nd.numOfLayers-1][0] == 1):
                self.setExpectedOutput(np.array([testSet[i].expectedOutput]))
            else:
                self.setExpectedOutput(np.array(testSet[i].expectedOutput))
            
            self.forwardPass()
            
            # Check the accuracy after each forward pass.
            self.accuracyOfNN()    
                
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
#       print("Input Layer: \n", nd.layerActivations[0])
#       print("Output Layer: \n", outputLayer)
        
        # Send output values into derivative of activation function.
        deriveOutput = self.activationFunction(outputLayer, "sigmoid", True)
    
        # Multiply the derived values with the error for the output layer.
        derivAndErr = deriveOutput * errContribution
        
        # Transpose the array for easier matrix multiplication.
        transDerivAndErr = np.matrix.transpose(derivAndErr)
        
        # Multiply with hidden layer which is before the output layer in the structure.
        deltaWeightsHtoO = np.dot(transDerivAndErr, nd.layerActivations[nd.numOfLayers-2])
        
        # Transpose again to fix the alignment of values.
        deltaWeightsHtoO = np.matrix.transpose(deltaWeightsHtoO)
        
        # Stores the delta weight 
        nd.layerGradients[nd.numOfLayers-2] = deltaWeightsHtoO
        
        # Sums up the delta gradients for hidden to output
        nd.layerGradientsSums[nd.numOfLayers-2] += deltaWeightsHtoO
        
        # Start at 2, the layerWeights is 1 size less than the layerActivation list.
        for i in range(2, nd.numOfLayers):
            
            # The second counter used to store the error value.
            j = nd.numOfLayers
            
            # Transpose hidden to output weight matrix.
            hiddenWeightTrans = np.matrix.transpose(nd.layerWeights[nd.numOfLayers-i])
            
            # Calculates the error at the hidden layer using the weight connections of this layer and the next.
            errContributionHidden = np.dot(nd.layerErrors[nd.numOfLayers-i], hiddenWeightTrans)
            
            # Stores the hidden layer error.
            nd.layerErrors[nd.numOfLayers-j] =  errContributionHidden
            
            # Derive the hidden layer values.
            derivHidden = self.activationFunction(nd.layerActivations[nd.numOfLayers-i], nd.networkLayers[i-1][1], True)
            
            # Multiply the derivative of the hidden multiplied with the error of that hidden layer.
            hiddenDerivAndErr = derivHidden * errContributionHidden
        
            transHiddenDerivAndErr = np.matrix.transpose(hiddenDerivAndErr)
          
            deltaWeightsItoH = np.dot(transHiddenDerivAndErr, nd.layerActivations[nd.numOfLayers-j])
            
            deltaWeightsItoH = np.matrix.transpose(deltaWeightsItoH)
            
            # Store the hidden layers gradients. 
            nd.layerGradients[nd.numOfLayers-j] = deltaWeightsItoH
            
            # Adds the delta weights for each epoch.
            nd.layerGradientsSums[nd.numOfLayers-j] += deltaWeightsItoH
            
            j+=1
        
        # Updates the bias for both hidden and output.
        nd.layerBias[1] = nd.layerBias[1] + (nd.learningRate * hiddenDerivAndErr)
        nd.layerBias[2] = nd.layerBias[2] + (nd.learningRate * derivAndErr)
        
    def updateWeights(self):
        """ Used to update the weights of each layer after a certain learning technique. As well as applies momentum """
        
        if(nd.layerDeltaWeightsPrev[0] is None):
            # For each set of weights per layer, update them.
            for i in range(len(nd.layerWeights)):
                nd.layerWeights[i] = nd.layerWeights[i] - nd.learningRate * nd.layerGradients[i]
                # Set the momentum matrices for the next pass
                nd.layerDeltaWeightsPrev[i] = nd.layerGradients[i]
        else:
            for i in range(len(nd.layerWeights)):
                nd.layerWeights[i] = nd.layerWeights[i] - (nd.learningRate * nd.layerGradients[i] + nd.momentumAlpha * nd.layerDeltaWeightsPrev[i])
                # Set the current delta weights for use next time.
                nd.layerDeltaWeightsPrev[i] = nd.layerGradients[i]
        
    
    def rPropagation(self):
        """ The rProp algorithm"""
        npos = 1.2
        nneg = 0.5
        
        # Get the sign values of each current and previous delta for each connection matrix
        signDIToH = np.sign(nd.layerGradientsSums[0])
        signPrevDIToH = np.sign(nd.layerGradientsSumsPrev[0])
        
        signDHToO = np.sign(nd.layerGradientsSums[1])
        signPrevDHToO = np.sign(nd.layerGradientsSumsPrev[1])
        
        checkArrIToH = signDIToH * signPrevDIToH
        checkArrHToO = signDHToO * signPrevDHToO
        
        # Only updates the hidden to output layer weights
        for i in range(nd.layerWeights[1].shape[0]):
            for j in range(nd.layerWeights[1].shape[1]):
                if(checkArrHToO[i,j] == 1):
                    # Update the delta, before applying it
                    nd.layerRpropDeltas[1][i,j] = nd.layerRpropDeltas[1][i,j] * npos
                
                    # Decide what we need to do with delta
                    if(nd.layerGradientsSums[1][i,j] > 0):
                        
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] - nd.layerRpropDeltas[1][i,j]
                    
                    elif(nd.layerGradientsSums[1][i,j] < 0):
                        
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] + nd.layerRpropDeltas[1][i,j]
                        
                    else:
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j]
                
                elif(checkArrHToO[i,j] == -1):
                    # Update the delta
                    nd.layerRpropDeltas[1][i,j] = nd.layerRpropDeltas[1][i,j] * nneg
                    nd.layerGradientsSums[1][i,j] = 0
                    
                else:
                    if(nd.layerGradientsSums[1][i,j] > 0):
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] - nd.layerRpropDeltas[1][i,j]
                    
                    elif(nd.layerGradientsSums[1][i,j] < 0):
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] + nd.layerRpropDeltas[1][i,j]
                    
                    else:
                        nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j]
                        
        # Keep the delta's within specific range
        nd.layerRpropDeltas[1] = np.clip(nd.layerRpropDeltas[1], 0.000001, 50)
        
        # Updates the Input to Hidden layer weights
        for i in range(nd.layerWeights[0].shape[0]):
            for j in range(nd.layerWeights[0].shape[1]):    
                
                if(checkArrIToH[i, j] == 1):
                    nd.layerRpropDeltas[0][i, j] =  nd.layerRpropDeltas[0][i, j] * npos
                
                    if(nd.layerGradientsSums[0][i,j] > 0):
                        
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerRpropDeltas[0][i,j]
                    
                    elif(nd.layerGradientsSums[0][i,j] < 0):
                        
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] + nd.layerRpropDeltas[0][i,j]
                    
                    else:
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j]
                
                elif(checkArrIToH[i, j] == -1):
                    
                    nd.layerRpropDeltas[0][i,j] = nd.layerRpropDeltas[0][i,j] * nneg
                    
                    nd.layerGradientsSums[0][i,j] = 0
                
                else:
                    if(nd.layerGradientsSums[0][i,j] > 0):
                        
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerRpropDeltas[0][i,j]
                    
                    elif(nd.layerGradientsSums[0][i,j] < 0):   
                        
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerRpropDeltas[0][i,j]
                    
                    else:
                        nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j]
        
        # Keep the delta's within specific range
        nd.layerRpropDeltas[0] = np.clip(nd.layerRpropDeltas[0], 0.000001, 50)
        
                        
    def deltaBarDelta(self):
        """ Delta bar delta implementation """
        # The decay factor and growth amount k
        d = 0.20
        k = 0.0001
        
        # Get the sign values of each current and previous delta for each connection matrix
        signDIToH = np.sign(nd.layerGradientsSums[0])
        signPrevDIToH = np.sign(nd.layerGradientsSumsPrev[0])
        
        signDHToO = np.sign(nd.layerGradientsSums[1])
        signPrevDHToO = np.sign(nd.layerGradientsSumsPrev[1])
        
        checkArrIToH = signDIToH * signPrevDIToH
        checkArrHToO = signDHToO * signPrevDHToO
        
        # Only updates the hidden to output layer weights
        for i in range(nd.layerWeights[1].shape[0]):
            for j in range(nd.layerWeights[1].shape[1]):
                if(checkArrHToO[i, j] == 1):
                    # Add to the learning rate the k growth factor.
                    nd.layerLearningVals[1][i, j] = nd.layerLearningVals[1][i, j] + k
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[1][i, j] = nd.layerGradientsSums[1][i, j] * nd.layerLearningVals[1][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] - nd.layerGradientsSums[1][i, j]
                
                elif(checkArrHToO[i, j] == -1):
                    # Multiply to the learning rate the decay.
                    nd.layerLearningVals[1][i, j] = nd.layerLearningVals[1][i, j] * (1-d)
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[1][i, j] =  nd.layerGradientsSums[1][i, j] * nd.layerLearningVals[1][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] - nd.layerGradientsSums[1][i, j]
                else:
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[1][i, j] = nd.layerGradientsSums[1][i, j] * nd.layerLearningVals[1][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[1][i, j] = nd.layerWeights[1][i, j] - nd.layerGradientsSums[1][i, j]
        # Limit the learning rates
        nd.layerLearningVals[1] = np.clip(nd.layerLearningVals[1], 0.0001, 0.005)
        
        # Only updates the hidden to output layer weights
        for i in range(nd.layerWeights[0].shape[0]):
            for j in range(nd.layerWeights[0].shape[1]):
                if(checkArrIToH[i, j] == 1):
                    # Add to the learning rate the k growth factor.
                    nd.layerLearningVals[0][i, j] = nd.layerLearningVals[0][i, j] + k
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[0][i, j] = nd.layerGradientsSums[0][i, j] * nd.layerLearningVals[0][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerGradientsSums[0][i, j]
                
                elif(checkArrIToH[i, j] == -1):
                    # Multiply to the learning rate the decay.
                    nd.layerLearningVals[0][i, j] = nd.layerLearningVals[0][i, j] * (1-d)
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[0][i, j] =  nd.layerGradientsSums[0][i, j] * nd.layerLearningVals[0][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerGradientsSums[0][i, j]
                
                else:
                    # Multiply the current sum of the gradient.
                    nd.layerGradientsSums[0][i, j] = nd.layerGradientsSums[0][i, j] * nd.layerLearningVals[0][i, j]
                    # Subtract from the current weight
                    nd.layerWeights[0][i, j] = nd.layerWeights[0][i, j] - nd.layerGradientsSums[0][i, j]
        # Limit the learning rates
        nd.layerLearningVals[0] = np.clip(nd.layerLearningVals[0], 0.0001, 0.005)
                   
        
    def setSumOfPrevGradients(self):
        """ Saves the current gradient sums to for the next pass. """
        
        for i in range(len(nd.layerGradientsSums)):
            # Sets the sums for next pass
            nd.layerGradientsSumsPrev[i] = np.copy(nd.layerGradientsSums[i])
            
            # Reset the gradient sum arrays
            nd.layerGradientsSums[i].fill(0)
#         print("Layer Gradients Sums Prev: \n", nd.layerGradientsSumsPrev)
#         print("Layer Gradients Sums Set to 0: \n", nd.layerGradientsSums)
            
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
            elif(outputLayer[0, i] < 0.5):
                outputLayer[0, i] = 0
        
        # Compare the current output to the expected to check accuracy.
        if(np.array_equal(outputLayer, nd.layerOutputTarget)):
            self.accuracyCount+=1
    
    def accuracyOfEpoch(self):
        correct = self.accuracyCount
        # Reset the count
        self.accuracyCount = 0
        return correct   
        
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