import numpy as np
import Utils as utils
import Smoothing as sm

# codifica dello pseudocodice dell'algoritmo di Viterbi
def viterbiAlgorithm(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary, smoothingVector):
    sentenceList = utils.tokenizeSentence(sentence)
    # convertiamo i valori delle strutture create in precedenza coi log
    logTPM = utils.matrixToLogMatrix(transitionProbabilityMatrix)
    logEPD = utils.dictToLogDict(emissionProbabilityDictionary)
    logSV = utils.matrixToLogMatrix(smoothingVector)

    viterbi = np.ones((len(pos), len(sentenceList)))
    backpointer = np.zeros((len(pos), len(sentenceList)))
    for index,p in enumerate(pos):
        if sentenceList[0] in emissionProbabilityDictionary.keys():
            viterbi[index, 0] = logTPM[0, index] + logEPD[sentenceList[0]][index]
        else:
            viterbi[index, 0] = logTPM[0, index] + logSV[index]
    for t,word in enumerate(sentenceList):
        for s,p in enumerate(pos):
            if word in emissionProbabilityDictionary.keys():
                viterbi[s, t] = np.max(viterbi[:, t-1] + logTPM[:, s] + logEPD[sentenceList[t]][s])
            else:
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + logSV[s])

            backpointer[s, t] = np.argmax(viterbi[:, t - 1] + logTPM[:, s])

    # calcolo del percorso migliore usando il backpointer
    bestPath = np.zeros(len(sentenceList))
    bestPath[len(sentenceList) - 1] =  viterbi[:, -1].argmax() # last state
    for t in range(len(sentenceList)-1, 0, -1): # states of (last-1)th to 0th time step
        bestPath[t-1] = backpointer[int(bestPath[t]),t]
        
    # stampa della parola con il tag corrispondente
    #for index,w in enumerate(sentenceList):
        #if w in temp:
           # print(sentenceList[index], pos[int(bestPath[index])])
    bestPos = []
    for bp in bestPath:
        bestPos.append(pos[int(bp)])

    return bestPos

def syntaxBasedDecoding(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary, language):
    sentenceList = utils.tokenizeSentence(sentence)
    # convertiamo i valori delle strutture create in precedenza coi log
    logTPM = utils.matrixToLogMatrix(transitionProbabilityMatrix)
    logEPD = utils.dictToLogDict(emissionProbabilityDictionary)

    viterbi = np.ones((len(pos), len(sentenceList)))
    backpointer = np.zeros((len(pos), len(sentenceList)))
    for index, p in enumerate(pos):
        if sentenceList[0] in emissionProbabilityDictionary.keys():
            viterbi[index, 0] = logTPM[0, index] + logEPD[sentenceList[0]][index]
        else:
            temp = utils.matrixToLogMatrix(sm.getUnknownTag(sentenceList[0], language, pos))
            viterbi[index, 0] = logTPM[0, index] + temp[index]
    for t, word in enumerate(sentenceList):
        for s, p in enumerate(pos):
            if word in emissionProbabilityDictionary.keys():
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + logEPD[sentenceList[t]][s])
            else:
                temp = utils.matrixToLogMatrix(sm.getUnknownTag(word, language, pos))
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + temp[s])

            backpointer[s, t] = np.argmax(viterbi[:, t - 1] + logTPM[:, s])

    # calcolo del percorso migliore usando il backpointer
    bestPath = np.zeros(len(sentenceList))
    bestPath[len(sentenceList) - 1] = viterbi[:, -1].argmax()  # last state
    for t in range(len(sentenceList) - 1, 0, -1):  # states of (last-1)th to 0th time step
        bestPath[t - 1] = backpointer[int(bestPath[t]), t]

    bestPos = []
    for bp in bestPath:
        bestPos.append(pos[int(bp)])

    return bestPos