# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

#troviamo tutti i pos del treebank
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
        if line.startswith('#'):
            analyze = False
        # analizziamo la riga e segnamo l'occorrenza del pos
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
        if line.startswith('#'):
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
            
def viterbiAlgorithm(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary):
    sentenceList = tokenizeSentence(sentence)
        
    viterbi = np.ones((len(pos), len(sentenceList)))
    backpointer = np.zeros((len(pos), len(sentenceList)))
    for index,p in enumerate(pos):
        viterbi[index, 0] = transitionProbabilityMatrix[0, index] * emissionProbabilityDictionary[sentenceList[0]][index]
        #backpointer[index, 0] = 0 #assegnazione superflua in quanto fatta con l'inizializzazione
    for t,word in enumerate(sentenceList[1::]):
        for s,p in enumerate(pos):
            viterbi[s, t] = np.max(viterbi[:, t-1] * transitionProbabilityMatrix[:, s] * emissionProbabilityDictionary[sentenceList[t]][s])
            backpointer[s, t] = np.argmax(viterbi[:, t-1] * transitionProbabilityMatrix[:, s])
    viterbi[len(pos)-1, len(sentenceList)-1] = np.max(viterbi[:, len(sentenceList)-1] * transitionProbabilityMatrix[:, len(pos)-1])
    backpointer[len(pos)-1, len(sentenceList)-1] = np.argmax(viterbi[:, len(sentenceList)-1] * transitionProbabilityMatrix[:, len(pos)-1])
    
    #print(viterbi)
    #print(backpointer)
    
    best_path = np.zeros(len(sentenceList))
    for index, c in enumerate(viterbi.T):
        best_path[index] = backpointer[np.argmax(viterbi[:, index])][index]
        
    for index,p in enumerate(pos):
        print(sentenceList[index], pos[int(best_path[index+1%len(pos)])])
        
    #print(best_path)
        


#-------------------------------------------------------
sentence = '+ Ego Andreas notarius, rogatus a Teuperto diacono, me teste subscripsi.'
fileName = 'la_llct-ud-train.conllu'
with open(fileName) as file:
    pos = findingAllPos(file)
    #print(pos) # stampa la lista di pos che compaiono nel treebank
    transitionProbabilityMatrix, emissionProbabilityDictionary = learningPhase(file, pos)
    viterbiAlgorithm(sentence, pos, transitionProbabilityMatrix, emissionProbabilityDictionary)
    print(pos)
