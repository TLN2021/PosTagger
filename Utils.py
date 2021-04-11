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

def tokenizeSentence(sentence):
    punctuations = '''!(){};:'"\,<>./?@#$%^&*_~'''
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
