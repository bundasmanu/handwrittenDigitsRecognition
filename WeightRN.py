
class WeightRN(object):

    def __init__(self):
        self.inputWeights=None
        self.outputWeights=None

    def getInputWeights(self):
        return self.inputWeights

    def setInputWeights(self, newInputWeights):
        self.inputWeights=newInputWeights

    def getOutputWeights(self):
        return self.outputWeights

    def setOutputWeights(self, newOutputWeights):
        self.outputWeights=newOutputWeights
