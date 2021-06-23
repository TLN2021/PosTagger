import numpy as np
import Utils as utils
import Evaluation as ev
import pandas as pd
import time

# fase di learning in cui si impara per ogni parola qual è il suo
# tag più frequente che compare nel corpus
def learningPhase(file, pos):
        mostFrequentTag = {}

        file.seek(0)  # usiamo la funzione per settare nuovamente il puntatore al file all'inizio
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
            # analizziamo la riga e segnamo l'occorrenza del pos
            if analyze is True:
                if (wordsInLine != []):
                    for index, p in enumerate(pos):
                        if p in wordsInLine:
                            temporaryEmissionArray = np.zeros(len(pos))
                            temporaryEmissionArray[index] += 1
                            if wordsInLine[1] in mostFrequentTag.keys():
                                mostFrequentTag[wordsInLine[1]] += temporaryEmissionArray
                            else:
                                mostFrequentTag[wordsInLine[1]] = temporaryEmissionArray

        for word in mostFrequentTag.keys():
            mostFrequentTag[word] = pos[np.argmax(list(mostFrequentTag[word]))]

        return mostFrequentTag

# fase di decoding a cui si assegna ad ogni parola del test set 
# il tag più frequente "imparato" per quel termine nel passo precedente
# nel caso in cui quella parola non compaia nel train set si assegna
# automaticamente il pos 'NOUN'
def decodingPhase(sentence, mostFrequentTag, smoothingVector):
    bestPos=[]
    sentenceList = utils.tokenizeSentence(sentence)
    for word in sentenceList:
        if word in mostFrequentTag.keys():
            bestPos.append(mostFrequentTag[word])
        else:
            bestPos.append('NOUN')
    return bestPos


# main della baseline
def simpleBaseLine(trainFileName, testFileName):
    startTime = time.time()
    # 1) LEARNING (sul training set)
    with open(trainFileName, 'r', encoding='utf-8') as trainFile:
        posInTrain = utils.findingAllPos(trainFile)
        mostFrequentTag = learningPhase(trainFile, posInTrain)

    learningTime = time.time() - startTime
    print("Learning time:", learningTime)

    decodingTimeStart = time.time()

    with open(testFileName, 'r', encoding='utf-8'):
        sentenceTest, correctPos = ev.getSencencePos(testFileName)

    # 2) DECODING (sul test set)
    baseLinePos = []
    for sentence in sentenceTest:
        baseLinePos.append(decodingPhase(sentence, mostFrequentTag, None))

    decodingTime = time.time() - decodingTimeStart
    print("Decoding time:", decodingTime)
    
    accuracyOnTest, errorVector = ev.accuracy(correctPos, baseLinePos)
    print("Accuracy Baseline:", accuracyOnTest)
    print("Errori più comuni:", pd.DataFrame(errorVector))

    return baseLinePos

trainSetFile = '../TreeBank - Latino/la_llct-ud-train.conllu'
testSetFile = '../TreeBank - Latino/la_llct-ud-test.conllu'
#trainSetFile = '../TreeBank - Greco/grc_perseus-ud-train.conllu'
#testSetFile = '../TreeBank - Greco/grc_perseus-ud-test.conllu'
simpleBaseLine(trainSetFile, testSetFile)




