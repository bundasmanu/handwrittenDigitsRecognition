import Utils as ut
import WeightRN as wr, Bias as bi
import numpy as np

class Network(object):

    def __init__(self, **netWorkData):
        self.inputs=netWorkData.get(ut.Utils.INPUTS)
        self.hiddenLayers=netWorkData.get(ut.Utils.HIDDEN_LAYERS)
        self.outputs=netWorkData.get(ut.Utils.OUTPUTS)
        self.bias= bi.Bias()
        self.weighss= wr.WeightRN()

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

    #Metodo que estabelece o algoritmo de propagação direta, aplicado a cada uma das particulas existentes, sendo que
    #retorna a perda de probabilidade logarítmica
    def forwardPropagation(self,dimensions,dataToLearn, targets):

        '''
        Depois de definidos os inputs:
            - Inputs: 64 -->representa os 64 pixeis, referentes a cada uma das imagens (0-16 valor possível em cada pixel)
            - Hidden Layers --> Camadas escondidas, pode se brincar com este valor, por exemplo o valor 20
            - Outputs: 10 --> Representa a classificacao possível da imagem: [0,1,2,3,4,5,6,7,8,9]
        :return:
            - Perda
        '''

        self.weighss.inputWeights= dimensions[:(self.inputs*self.hiddenLayers)].reshape(self.inputs,self.hiddenLayers)
        self.bias.inputBias = dimensions[(self.inputs*self.hiddenLayers):((self.inputs*self.hiddenLayers)+self.hiddenLayers)].reshape(self.hiddenLayers,)
        self.weighss.outputWeights= dimensions[((self.inputs*self.hiddenLayers)+self.hiddenLayers):(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs))].reshape(self.hiddenLayers,self.outputs)
        self.bias.outputBias= dimensions[(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs)):(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs))+self.outputs].reshape(self.outputs,)

        '''
        Definicao agora do algoritmo forward propagation, mediante a logica,
        que visualizei nos artigos que visualizei
        '''
        z1= dataToLearn.dot(self.weighss.inputWeights) + self.bias.inputBias #Pre Ativacao Camada 1
        a1= np.tanh(z1) # Ativacao na Camada 1
        z2= a1.dot(self.weighss.outputWeights)+ self.bias.outputBias #Pre Ativacao Camada 2
        logits =z2 # Logits para a Camada 2

        '''
        Computacao do softmax do Logits
        '''
        expScores= np.exp(logits)
        probs= expScores / np.sum(expScores, axis=1, keepdims=True)

        '''
        Calcular a probabilidade de log negativo
        '''
        correctLogProbs= -np.log(probs[range(1797), targets]) #1797 corresponde ao nº de imagens em analise,
        loss = np.sum(correctLogProbs)/ 1797

        return loss

    def aplicarFuncaoObjetivoTodasParticulas(self,arrayParticulasDimensao,dataToLearn, targets):

        '''
            Input: Array Bidimensional--> contendo para cada particula, a posicao de cada uma destas, tendo em conta o nº de dimensoes existentes
        :return:
            Retorno: Array Unidimensional --> Contendo a perda resultante do calculo da funcao objetivo, para cada uma das particulas
        '''

        numberParticles= arrayParticulasDimensao.shape[0] #--> Numero de particulas, visto que o arrayParticulasDimensao[Particulas][Dimensoes] é constituido desta forma, o shape[0] retorna o numero de particulas existentes

        lossOfEveryParticle= [self.forwardPropagation(arrayParticulasDimensao[i],dataToLearn,targets) for i in range(numberParticles)] #-->A utilizacao do range cria uma sequencia ordenada de valores, dentro da gama indicada, em vez de utilizar i++
        return lossOfEveryParticle #--> Retorno do Array Unidimensional, que contem a perda resultante da aplicacao da funcao objetivo (forward propagation), a cada uma das particulas

    def predict(self,dataToLearn,arrayPositionsOfParticles):

        '''

        :param dataToLearn: --> digitos.data, corresponde aos dados de aprendizagem
        :param arrayPositionsOfParticles: --> vetor unidimensional, contendo a posicao otima encontrada pelo swarm, para cada uma das dimensoes existentes, se forem 163, contem um vetor unidimensional, com 1 linha e 163 colunas, correspondendo cada coluna à posicao otima de cada dimensao
        :return:
            Array Unidimensional --> com as previsoes ao longo do eixo das linhas, retornando o maior valor presente em cada uma das linhas, que se encontra no array bidimensional logits
        '''

        self.weighss.inputWeights= arrayPositionsOfParticles[:(self.inputs*self.hiddenLayers)].reshape(self.inputs,self.hiddenLayers)
        self.bias.inputBias = arrayPositionsOfParticles[(self.inputs*self.hiddenLayers):((self.inputs*self.hiddenLayers)+self.hiddenLayers)].reshape(self.hiddenLayers,)
        self.weighss.outputWeights= arrayPositionsOfParticles[((self.inputs*self.hiddenLayers)+self.hiddenLayers),(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs))].reshape(self.hiddenLayers,self.outputs)
        self.bias.outputBias= arrayPositionsOfParticles[(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs)),(((self.inputs*self.hiddenLayers)+self.hiddenLayers)+(self.hiddenLayers*self.outputs))+self.outputs].reshape(self.outputs,)

        '''
        Definicao agora do algoritmo forward propagation, mediante a logica,
        que visualizei nos artigos que visualizei
        '''
        z1 = dataToLearn.dot(self.weighss.inputWeights) + self.bias.inputBias  # Pre Ativacao Camada 1
        a1 = np.tanh(z1)  # Ativacao na Camada 1
        z2 = a1.dot(self.weighss.outputWeights) + self.bias.outputBias  # Pre Ativacao Camada 2
        logits = z2  # Logits para a Camada 2

        y_pred= np.argmax(logits,axis=1) #Array Unidimensional --> com as previsoes ao longo do eixo das linhas, retornando o maior valor presente em cada uma das linhas, que se encontra no array bidimensional logits
        return y_pred