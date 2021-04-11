import numpy as np

def learningPhase(file, pos):
    transitionProbabilityMatrix = np.zeros(
        (len(pos), len(pos)))  # matrice posXpos che contiene i conteggi delle coppie tagPrecedente-tagCorrente
    countPos = np.zeros(len(
        pos))  # vettore che contiene il conteggio dei pos del file dove l'indice corrisponde a quello del vettore "pos"
    previousPosIndex = -1
    emissionProbabilityDictionary = {}

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
            previousPosIndex = -1
        # analizziamo la riga e segnamo l'occorrenza del pos
        if analyze is True:
            if (wordsInLine != []):
                for index, p in enumerate(pos):
                    if p in wordsInLine:
                        temporaryEmissionArray = np.zeros(len(pos))
                        temporaryEmissionArray[index] += 1
                        if wordsInLine[1] in emissionProbabilityDictionary.keys():
                            emissionProbabilityDictionary[wordsInLine[1]] += temporaryEmissionArray
                        else:
                            emissionProbabilityDictionary[wordsInLine[1]] = temporaryEmissionArray

                        # controlliamo che stiamo analizzando una nuova frase
                        if (previousPosIndex >= 0):
                            # aggiorniamo la matrice del conteggio della transizione
                            transitionProbabilityMatrix[previousPosIndex][index] += 1
                        # aggiorniamo il conteggio per il pos corrente e ci salviamo l'indece per l'iterata successiva
                        countPos[index] += 1
                        previousPosIndex = index

    # print(countPos)  # stampa le occorrenze di ogni pos nel treebank

    # stampa la somma delle volte in cui ogni pos precede gli altri
    # (dovrebbe risultare somma <= occorrenze del pos )
    # for row in transitionProbabilityMatrix:
    #    print(np.sum(row)) # somma tutti i valori nella riga
    # print(np.sum(transitionProbabilityMatrix[len(transitionProbabilityMatrix)-1])) # somma dei valori nell'ultima riga della matrice

    # calcola la probabilità facendo conteggio_coppia/conteggio_tag_precedente
    # nota : le 2 righe successive sono equivalenti
    # transitionProbabilityMatrix = transitionProbabilityMatrix / countPos[:, None]
    transitionProbabilityMatrix /= countPos.reshape(-1, 1)

    for key in emissionProbabilityDictionary.keys():
        emissionProbabilityDictionary[key] /= countPos

    return transitionProbabilityMatrix, emissionProbabilityDictionary