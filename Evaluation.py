import numpy as np
from sklearn.metrics import accuracy_score

# restituisce le frasi nel treebank ed i relativi pos
def getSencencePos(fileName):
    with open(fileName, 'r', encoding='utf-8') as file:
        analyze = False
        lines = file.readlines();
        sentenceIndex = -1  # indice che conta le frasi nel testo
        sentences = []
        pos = {}
        for line in lines:
            wordsInLine = line.split()
            if line.startswith('#') or wordsInLine == []:
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

def accuracy (target,target_test):
    accuracy=[]
    for index in range(len(target)):
        accuracy.append(accuracy_score(target[index], target_test[index]))
    return np.mean(accuracy)