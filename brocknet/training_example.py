'''
Created on Apr 23, 2017

@author: Matthew Kirchhof, Imran Qureshi

The object that holds a single training examples input and expected output
'''

class TrainingExample:
    """Used to hold each data piece with its respected expected result"""
        
    inputData = None
    expectedOutput = None
    
    def __init__(self, idata, eo):
        self.inputData = idata
        self.expectedOutput = eo
        