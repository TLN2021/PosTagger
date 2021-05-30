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
    # considera il pos 'start'
    for index,p in enumerate(pos):
        if sentenceList[0].lower() in emissionProbabilityDictionary.keys():
            viterbi[index, 0] = logTPM[len(pos)-2, index] + logEPD[sentenceList[0].lower()][index]
        else:
            viterbi[index, 0] = logTPM[len(pos)-2, index] + logSV[index]
            
    for t,word in enumerate(sentenceList):
        for s,p in enumerate(pos):
            if word.lower() in emissionProbabilityDictionary.keys():
                viterbi[s, t] = np.max(viterbi[:, t-1] + logTPM[:, s] + logEPD[sentenceList[t].lower()][s])
            else:
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + logSV[s])

            backpointer[s, t] = np.argmax(viterbi[:, t - 1] + logTPM[:, s])

    # considera il pos 'end'
    viterbi[len(pos)-1, len(sentenceList)-1] = np.max(viterbi[:, len(sentenceList)-1] + logTPM[:, len(pos)-1] )
    backpointer[len(pos)-1, len(sentenceList)-1] = np.argmax(viterbi[:,  len(sentenceList)-1] + logTPM[:, len(pos)-1])

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

# tipologia di decoding che utilizza come smothing u approccio basato sui suffissi di nomi, aggettivi e verbi in base alla lingua
def syntaxBasedDecoding(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary, language):
    sentenceList = utils.tokenizeSentence(sentence)
    # convertiamo i valori delle strutture create in precedenza coi log
    logTPM = utils.matrixToLogMatrix(transitionProbabilityMatrix)
    logEPD = utils.dictToLogDict(emissionProbabilityDictionary)

    viterbi = np.ones((len(pos), len(sentenceList)))
    backpointer = np.zeros((len(pos), len(sentenceList)))
    # considera il pos 'start'
    for index, p in enumerate(pos):
        if sentenceList[0].lower() in emissionProbabilityDictionary.keys():
            viterbi[index, 0] = logTPM[len(pos)-2, index] + logEPD[sentenceList[0].lower()][index]
        else:
            temp = utils.matrixToLogMatrix(sm.getUnknownTag(sentenceList[0], language, pos))
            viterbi[index, 0] = logTPM[len(pos)-2, index] + temp[index]
    for t, word in enumerate(sentenceList):
        for s, p in enumerate(pos):
            if word.lower() in emissionProbabilityDictionary.keys():
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + logEPD[sentenceList[t].lower()][s])
            else:
                temp = utils.matrixToLogMatrix(sm.getUnknownTag(word.lower(), language, pos))
                viterbi[s, t] = np.max(viterbi[:, t - 1] + logTPM[:, s] + temp[s])

            backpointer[s, t] = np.argmax(viterbi[:, t - 1] + logTPM[:, s])

    # considera il pos 'end'
    viterbi[len(pos) - 1, len(sentenceList) - 1] = np.max(viterbi[:, len(sentenceList) - 1] + logTPM[:, len(pos) - 1])
    backpointer[len(pos) - 1, len(sentenceList) - 1] = np.argmax(viterbi[:, len(sentenceList) - 1] + logTPM[:, len(pos) - 1])

    # calcolo del percorso migliore usando il backpointer
    bestPath = np.zeros(len(sentenceList))
    bestPath[len(sentenceList) - 1] = viterbi[:, -1].argmax()  # last state
    for t in range(len(sentenceList) - 1, 0, -1):  # states of (last-1)th to 0th time step
        bestPath[t - 1] = backpointer[int(bestPath[t]), t]

    bestPos = []
    for bp in bestPath:
        bestPos.append(pos[int(bp)])

    return bestPos