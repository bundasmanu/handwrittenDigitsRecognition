import random

class Utils:

    INPUTS='nInputs'
    HIDDEN_LAYERS='nHiddenLayers'
    OUTPUTS='nOutputs'

    def generateRandomValue(self,minLimit,maxLimit):
        return round(random.uniform(minLimit,maxLimit),2)