from sklearn import datasets,svm, metrics
import matplotlib.pyplot as plt

#Obtencao do Dataset referente aos problema do reconhecimento de Digitos, sendo estes entre [0-9]
global digitos, classifier
digitos=datasets.load_digits()
classifier=svm.SVC(gamma=0.01) #gamma, corresponde a 1/nº de features, neste caso (64), nº de pixeis, parametros no website do scikit-learn svm.SVC

#tentativa inicial de listar dados referentes ao dataset para perceber melhor, o que se pretende
def listagemDadosDigitos(digits):
    print(dir(digits)) #--> atributos do objeto digits, sendo estes: ['DESCR', 'data', 'images', 'target', 'target_names']
    print(digits.images)
    print(digits.target)
    print(digits.target_names)#--> [0 1 2 3 4 5 6 7 8 9]
    print(digits.images.shape)
    print(digits.target.shape)

'''Atraves dos prints efetuados atras, percebi
que o dataset é composto por 1797 entradas (imagens), sendo
que cada um destes é representado por uma matriz 8*8'''

def printFirstImage(digits): #-->Atraves deste print percebo que para a 1ª imagem das 1797, existe uma matriz bidimensional, que representa um pixel, e o seu valor, e a quantidade de preto representada no pixel
    print(digits)
    plt.imshow(digits,cmap='binary')
    plt.show()#-->No SciView conseguimos perceber que esta imagem representa o número 0

def transformBitoUniMatrix(digits): #-->Objetivo transformar matriz 8*8 em 1*64
    return digits.images.reshape(len(digits.images),64)#-->Transformacao em 1 linha com 64 colunas, podia ter colocado -1, assim assumia que tinha apenas uma linha e preenchia automaticamente

#-->De modo a reduzir a carga computacional, e ainda de modo a perceber se aplicando um treino de 50%, este revela-se ou não eficaz
def trainingApp(digitsUnidimensional):

    '''
    Das 1797 imagens, vou treinar apenas a 1ª metade (1797/2), e depois vou tentar prever a 2 parte, para ver se resultou conforme o esperado
    '''
    classifier.fit(digitsUnidimensional[:int(len(digitos.images) / 2)], digitos.target[:int(len(digitos.target) / 2)])
    #classifier.fit(digitsUnidimensional[:],digitos.target[:])#-->Treino total

#-->Agora vou testar apenas o 2 lado que nao foi treinado
def testApp(digitsUnidimensional):

    result=classifier.predict(digitsUnidimensional[int(len(digitos.images)/2):])
    return result;

def main():
    listagemDadosDigitos(digitos)
    printFirstImage(digitos.images[0])
    transformDigits = transformBitoUniMatrix(digitos)
    print(transformDigits[0][10])#-->Verificar se aplicou bem a reducao
    trainingApp(transformDigits)
    print("\n\n------------------------Resultados:\n")
    resultadoExpectavel=digitos.target[int(len(digitos.target)/2):]
    resultadosTreino=testApp(transformDigits)
    result=metrics.classification.accuracy_score(resultadoExpectavel,resultadosTreino)
    xpto=metrics.classification_report(resultadoExpectavel,resultadosTreino)
    print("Resultado final: ",result*100,"\n")
    print(xpto)
    print("-------------------------------")

if __name__ == "__main__":
    main()