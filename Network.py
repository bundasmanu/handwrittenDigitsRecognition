import Utils as ut
import Weight
import Bias

class Network:

    '''
    Propriedades da classe
    '''
    weights = Weight()
    bias= Bias()

    def __init__(self, **netWorkData):
        self.inputs=netWorkData.get(ut.Utils.INPUTS)
        self.hiddenLayers=netWorkData.get(ut.Utils.HIDDEN_LAYERS)
        self.outputs=netWorkData.get(ut.Utils.OUTPUTS)

    def getInputs(self):
        return self.inputs

    def setInputs(self,newNumberInputs):
        self.inputs=newNumberInputs

    def getHiddenLayers(self):
        return self.hiddenLayers

    def setHiddenLayers(self,newNumberHiddenLayers):
        self.hiddenLayers=newNumberHiddenLayers

    def getOutputs(self):
        return self.outputs

    def setOuputs(self,newNumberOutputs):
        self.outputs=newNumberOutputs
