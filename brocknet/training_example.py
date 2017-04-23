'''
Created on Apr 23, 2017

@author: Matth
'''

class TrainingExample():
        
    inputData = None
    expectedOutput = None
    
    def __init__(self,idata,eo):
        self.inputData = idata
        self.expectedOutput = eo
        