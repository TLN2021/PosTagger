import numpy as np
import Utils as utils
import Smoothing as sm
import Evaluation as ev

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

def decodingPhase(sentence, mostFrequentTag, smoothingVector):
    bestPos=[]
    sentenceList = utils.tokenizeSentence(sentence)
    for word in sentenceList:
        if word in mostFrequentTag.keys():
            bestPos.append(mostFrequentTag[word])
        else:
            bestPos.append(np.max(smoothingVector))
    return bestPos



def simpleBaseLine(trainFileName, testFileName):

    # 1) LEARNING (sul training set)
    with open(trainFileName, 'r', encoding='utf-8') as trainFile:
        posInTrain = utils.findingAllPos(trainFile)
        mostFrequentTag = learningPhase(trainFile, posInTrain)

    with open(testFileName, 'r', encoding='utf-8') as testFile:
        sentenceTest, correctPos = ev.getSencencePos(testFileName)

    # 1.5) SMOOTHING
    smoothingType = 0
    smoothingVector = sm.smoothing(posInTrain, smoothingType, None)

    # 2) DECODING (sul test set)
    baseLinePos = []
    for sentence in sentenceTest:
        baseLinePos.append(decodingPhase(sentence, mostFrequentTag, smoothingVector))

    accuracyOnTest = ev.accuracy(correctPos, baseLinePos)
    print(accuracyOnTest)

    return baseLinePos

#train_set_file = 'TreeBank - Latino/la_llct-ud-train.conllu'
#testSetFile = 'TreeBank - Latino/la_llct-ud-test.conllu'
trainSetFile = 'TreeBank - Greco/grc_perseus-ud-train.conllu'
testSetFile = 'TreeBank - Greco/grc_perseus-ud-test.conllu'
print(simpleBaseLine(trainSetFile, testSetFile))




