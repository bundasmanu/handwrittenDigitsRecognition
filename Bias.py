
class Bias(object):

    def __init__(self):
        self.inputBias = None
        self.outputBias = None

    def getInputBias(self):
        return self.inputBias

    def setInputBias(self, newInputBias):
        self.inputBias=newInputBias

    def getOutputBias(self):
        return self.outputBias

    def setOutputBias(self, newOutputBias):
        self.outputBias=newOutputBias
