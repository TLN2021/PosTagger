# -*- coding: utf-8 -*-
import numpy as np

# restituisce tutti i pos del treebank
def findingAllPos(file):
    pos = []
    file.seek(0) #usiamo la funzione per settare nuovamente il puntatore al file all'inizio
    lines = file.readlines();
    for line in lines:
        wordsInLine = line.split()
        # i seguenti due controlli servono per capire quando analizzare il corpus per il pos
        # Se la riga incomincia con '#' significa che è iniziato il corpus successivo 
        # e c'è bisogno di resettare il tag precedente
        if line.startswith('1'):
            analyze = True
        elif line.startswith('#'):
            analyze = False
        # analizziamo la riga e segniamo l'occorrenza del pos
        if analyze is True:
            if(wordsInLine != []):
                if(wordsInLine[3] not in pos):
                    pos.append(wordsInLine[3])
    return pos

def learningPhase(file, pos):
    transitionProbabilityMatrix = np.zeros( (len(pos), len(pos)) ) # matrice posXpos che contiene i conteggi delle coppie tagPrecedente-tagCorrente
    countPos = np.zeros(len(pos)) # vettore che contiene il conteggio dei pos del file dove l'indice corrisponde a quello del vettore "pos"
    previousPosIndex = -1
    emissionProbabilityDictionary = {}
    
    file.seek(0) #usiamo la funzione per settare nuovamente il puntatore al file all'inizio
    lines = file.readlines();
    for line in lines:
        wordsInLine = line.split()
        # i seguenti due controlli servono per capire quando analizzare il corpus per il pos
        # Se la riga incomincia con '#' significa che è iniziato il corpus successivo 
        # e c'è bisogno di resettare il tag precedente
        if line.startswith('1'):
            analyze = True
        elif line.startswith('#'):
            analyze = False
            previousPosIndex = -1
        # analizziamo la riga e segnamo l'occorrenza del pos
        if analyze is True:
            if(wordsInLine != []):
                for index,p in enumerate(pos):  
                    if p in wordsInLine:
                        temporaryEmissionArray = np.zeros(len(pos))
                        temporaryEmissionArray[index] += 1
                        if wordsInLine[1] in emissionProbabilityDictionary.keys():
                            emissionProbabilityDictionary[wordsInLine[1]] += temporaryEmissionArray
                        else:
                            emissionProbabilityDictionary[wordsInLine[1]] = temporaryEmissionArray
                            
                        # controlliamo che stiamo analizzando una nuova frase
                        if(previousPosIndex >= 0):
                            # aggiorniamo la matrice del conteggio della transizione
                            transitionProbabilityMatrix[previousPosIndex][index] += 1
                        #aggiorniamo il conteggio per il pos corrente e ci salviamo l'indece per l'iterata successiva
                        countPos[index] += 1 
                        previousPosIndex = index

    #print(countPos)  # stampa le occorrenze di ogni pos nel treebank
    
    # stampa la somma delle volte in cui ogni pos precede gli altri
    # (dovrebbe risultare somma <= occorrenze del pos )
    #for row in transitionProbabilityMatrix:
    #    print(np.sum(row)) # somma tutti i valori nella riga
    #print(np.sum(transitionProbabilityMatrix[len(transitionProbabilityMatrix)-1])) # somma dei valori nell'ultima riga della matrice

    # calcola la probabilità facendo conteggio_coppia/conteggio_tag_precedente
    # nota : le 2 righe successive sono equivalenti
    #transitionProbabilityMatrix = transitionProbabilityMatrix / countPos[:, None]
    transitionProbabilityMatrix /= countPos.reshape(-1,1) 
    
    for key in emissionProbabilityDictionary.keys():
        emissionProbabilityDictionary[key] /= countPos 
        
    return transitionProbabilityMatrix, emissionProbabilityDictionary

def tokenizeSentence(sentence):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in sentence:
        if char in punctuations:
            sentence = sentence.replace(char, ' ' + char)
    return sentence.split()

# restituisce una matrice i cui valori sono convertiti in log
def matrixToLogMatrix(matrix):
    tiny=np.finfo(0.).tiny #il più piccolo valore del compilatore py
    return np.log(matrix + tiny)

# restituisce un dizionario i cui valori associati alle chiavi sono convertiti in log
def dictToLogDict(dictionary):
    tiny=np.finfo(0.).tiny #il più piccolo valore del compilatore py
    newDict={}
    for key in dictionary.keys():
        newDict[key] = np.log(dictionary[key]+tiny)
    return newDict
            
# codifica dello pseudocodice dell'algoritmo di Viterbi
def viterbiAlgorithm(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary):
    sentenceList = tokenizeSentence(sentence)
    # convertiamo i valori delle strutture create in precedenza coi log
    logTPM = matrixToLogMatrix(transitionProbabilityMatrix)
    logEPD = dictToLogDict(emissionProbabilityDictionary)
        
    viterbi = np.ones((len(pos), len(sentenceList)))
    backpointer = np.zeros((len(pos), len(sentenceList)))
    for index,p in enumerate(pos):
        viterbi[index, 0] = logTPM[0, index] + logEPD[sentenceList[0]][index]
    for t,word in enumerate(sentenceList):
        for s,p in enumerate(pos):
            viterbi[s, t] = np.max(viterbi[:, t-1] + logTPM[:, s] + logEPD[sentenceList[t]][s])
            backpointer[s, t] = np.argmax(viterbi[:, t-1] + logTPM[:, s])

    # calcolo del percorso migliore usando il backpointer
    bestPath = np.zeros(len(sentenceList))
    bestPath[len(sentenceList) - 1] =  viterbi[:, -1].argmax() # last state
    for t in range(len(sentenceList)-1, 0, -1): # states of (last-1)th to 0th time step
        bestPath[t-1] = backpointer[int(bestPath[t]),t]
    # stampa della parola con il tag corrispondente
    for index,p in enumerate(sentenceList):
       print(sentenceList[index], pos[int(bestPath[index])])

    return bestPath


# restituisce le frasi nel treebank ed i relativi pos
def getSencencePos(fileName):
    with open(fileName, 'r', encoding='utf-8') as file:
        analyze= False
        lines = file.readlines();
        sentenceIndex=-1 # indice che conta le frasi nel testo
        sentences=[]
        pos={}
        for line in lines:
            wordsInLine = line.split()
            if line.startswith('#') or wordsInLine == [] :
                analyze = False
            elif line.startswith('1'): # si devono salvare i pos della frase
                analyze = True
            if 'text' in line :  # inizio di una nuova frase
                sentenceIndex += 1
                sentences.append(line.replace('# text = ','').replace('\n','')) # salva la frase
                pos[sentenceIndex]=[] # inizializza l'array per i pos della frase
            if analyze is True:
                pos[sentenceIndex].append(wordsInLine[3])
            
    return sentences , pos.values()


def accuracy (desiderato,ottenuto):
    accuracy=0
    return accuracy


def main(lingua):
            
    if lingua == "Latino":
        train_set_file = 'TreeBank - Latino/la_llct-ud-train.conllu'
        testSetFile = 'TreeBank - Latino/la_llct-ud-test.conllu'
    elif lingua == "Greco":
        train_set_file = 'TreeBank - Greco/grc_perseus-ud-train.conllu'
        testSetFile = 'TreeBank - Greco/grc_perseus-ud-test.conllu'

    # trovo per 
    sentenceTest, correctPos = getSencencePos(testSetFile)

    # 1) LEARNING (sul training set)
    with open(train_set_file, 'r', encoding='utf-8') as trainFile:
        posInTrain = findingAllPos(trainFile)
        transitionProbabilityMatrix, emissionProbabilityDictionary = learningPhase(trainFile, posInTrain)

    # 2) DECODING (sul test set)
    viterbiPos = []
    for sentence in sentenceTest:
        viterbiPos.append(viterbiAlgorithm(sentence, posInTrain, transitionProbabilityMatrix, emissionProbabilityDictionary))
        
    accuracyOnTest = accuracy(correctPos, viterbiPos)
    print(accuracyOnTest)


#-------------------------------------------------------

lingua= "Latino"
#lingua="Greco"
main(lingua)



