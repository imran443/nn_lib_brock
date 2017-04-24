# Used to hold each data piece with its respected expected result.
class TrainingExample:
        
    inputData = None
    expectedOutput = None
    
    def __init__(self, idata, eo):
        self.inputData = idata
        self.expectedOutput = eo
        