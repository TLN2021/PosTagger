# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import numpy as np

#troviamo tutti i pos del treebank
def findingAllPos(file):
    pos = []
    patterns= ['[A-Z]+']

    #troviamo nel file tutte le parole maiuscole
    text = file.read()
    for p in patterns:
        match = re.findall(p, text)
        for word in match:
            if(len(word) > 1 and word not in pos):
                pos.append(word) 
    #controlliamo che quelle parole non compaiano nelle frasi -> quindi siano effettivamente dei pos
    file.seek(0) #usiamo la funzione per settare nuovamente il puntatore al file all'inizio
    lines = file.readlines()
    for line in lines:
        wordsInLine = line.split()
        if 'text' in wordsInLine:
            for p in pos:
                if p in line:
                    pos.remove(p)
    return pos

def findingTransitionMatrix(file, pos):
    transitionMatrix = np.zeros( (len(pos), len(pos)) ) # matrice posXpos che contiene i conteggi delle coppie tagPrecedente-tagCorrente
    countPos = np.zeros(len(pos)) # vettore che contiene il conteggio dei pos del file dove l'indice corrisponde a quello del vettore "pos"
    previousPosIndex = -1
    
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
            for index,p in enumerate(pos):
                if p in wordsInLine:
                    # controlliamo che stiamo analizzando una nuova frase
                    if(previousPosIndex >= 0):
                        # aggiorniamo la matrice del conteggio della transizione
                        transitionMatrix[previousPosIndex][index] += 1
                    #aggiorniamo il conteggio per il pos corrente e ci salviamo l'indece per l'iterata successiva
                    countPos[index] += 1 
                    previousPosIndex = index
    #print(transitionMatrix)

    print(pos) # stampa la lista di pos che compaiono nel treebank
    print(countPos)  # stampa le occorrenze di ogni pos nel treebank
    # stampa la somma delle volte in cui ogni pos precede gli altri
    # (dovrebbe risultare somma <= occorrenze del pos )
    for row in transitionMatrix:
        print(np.sum(row)) # somma tutti i valori nella riga
    print(np.sum(transitionMatrix[len(transitionMatrix)-1])) # somma dei valori nell'ultima riga della matrice

    # calcola la probabilità facendo conteggio_coppia/conteggio_tag_precedente
    # nota : le 2 righe successive sono equivalenti
    #transitionMatrix = transitionMatrix / countPos[:, None]
    transitionMatrix = transitionMatrix / countPos.reshape(-1,1)
    print(transitionMatrix)
    for row in transitionMatrix:
        print(np.sum(row))

            

#-------------------------------------------------------
fileName = 'la_llct-ud-train.conllu'
with open(fileName) as file:
    pos = findingAllPos(file)
    findingTransitionMatrix(file, pos)
    #print(pos)


