'''
Created on Apr 23, 2017

@author: Matth
'''

class training_example():
        
    inputData = None
    expectedOutput = None
    
    def __init__(self,idata,eo):
        self.inputData = idata
        self.expectedOutput = eo
        