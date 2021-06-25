import numpy as np
from operator import itemgetter

# restituisce le frasi nel treebank ed i relativi pos
def getSencencePos(fileName):
    with open(fileName, 'r', encoding='utf-8') as file:
        analyze = False
        lines = file.readlines()
        sentenceIndex = -1  # indice che conta le frasi nel testo
        sentences = []
        pos = {}
        for line in lines:
            wordsInLine = line.split()
            if line.startswith('#') or wordsInLine == [] or wordsInLine == "":
                analyze = False
            elif line.startswith('1'):  # si devono salvare i pos della frase
                analyze = True
            if 'text' in line:  # inizio di una nuova frase
                sentenceIndex += 1
                sentences.append(line.replace('# text = ', '').replace('\n', ''))  # salva la frase
                pos[sentenceIndex] = []  # inizializza l'array per i pos della frase
            if analyze is True:
                pos[sentenceIndex].append(wordsInLine[3])
    return sentences, list(pos.values())

# restituisce l'accuracy confrontando il vettore target e quello prodotto + 
# la lista ordinata degli errori più frequenti
def accuracy (target,target_test):
    accuracy=[]
    errorPos = {}
    for index in range(len(target)):
        accuracyTemp = 0
        for i in range(0, len(target[index])):
            if target[index][i] == target_test[index][i]:
                accuracyTemp += 1
            else:
                if target[index][i] not in errorPos.keys():
                    errorPos[target[index][i]] = 0
                else:
                    errorPos[target[index][i]] += 1
        accuracy.append(accuracyTemp/len(target[index]))
        
    # memorizza gli errori ordinati per i più comuni dei pos
    errorVector = []
    for k, v in sorted(errorPos.items(), reverse=True, key=itemgetter(1)):
        errorVector.append(str(k) + " " + str(v))
    
    return np.mean(accuracy), errorVector
