import numpy as np

def statisticsOnDevSet(fileName, pos):
    with open(fileName, 'r', encoding='utf-8') as file:
        lines = file.readlines();
        words={}
        deletedWords=[]
        statistics = np.zeros(len(pos))
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
                if (wordsInLine != []):
                    w=wordsInLine[1]
                    if w not in words.keys() and w not in deletedWords:
                        words[w]=wordsInLine[3]
                    elif w in words.keys():
                        words.pop(w,None)
                        deletedWords.append(w)

        for index,p in enumerate(pos):
            statistics[index] = list(words.values()).count(p)

        return statistics/len(words.keys())


def smoothing(pos, type, devFileName) :
    smoothingVector=np.zeros(len(pos))
    if type == 0:
        for index,p in enumerate(pos) :
            if p=="NOUN":
                smoothingVector[index]=1
    elif type == 1:
        for index, p in enumerate(pos):
            if p == "NOUN" or p == "VERB":
                smoothingVector[index] = 0.5
    elif type == 2:
        smoothingVector = np.ones(len(pos))*(1/len(pos))
    elif type == 3:
        smoothingVector = statisticsOnDevSet(devFileName, pos)
    return smoothingVector