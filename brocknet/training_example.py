'''
Created on Apr 23, 2017

@author: Matthew Kirchhof, Imran Qureshi

The object that holds a single training examples input and expected output
'''

class TrainingExample():
        
    inputData = None
    expectedOutput = None
    
    def __init__(self,idata,eo):
        self.inputData = idata
        self.expectedOutput = eo
        