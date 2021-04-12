import numpy as np

def statisticsOnDevSet(fileName, pos):
    with open(fileName, 'r', encoding='utf-8') as file:
        lines = file.readlines()
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
                    w = wordsInLine[1]
                    if w not in words.keys() and w not in deletedWords:
                        words[w] = wordsInLine[3]
                    elif w in words.keys():
                        words.pop(w,None)
                        deletedWords.append(w)

        for index,p in enumerate(pos):
            statistics[index] = list(words.values()).count(p)

        return statistics/len(words.keys())


def smoothing(pos, type, devFileName) :
    smoothingVector = np.zeros(len(pos))
    if type == 0:
        smoothingVector[np.where(pos == "NOUN")] = 1
    elif type == 1:
        smoothingVector[np.where(pos == "NOUN")] = 0.5
        smoothingVector[np.where(pos == "VERB")] = 0.5
    elif type == 2:
        smoothingVector = np.ones(len(pos))*(1/len(pos))
    elif type == 3:
        smoothingVector = statisticsOnDevSet(devFileName, pos)
    return smoothingVector


def getUnknownTag(word, language, pos):
    posWord = np.zeros(len(pos))
    nounAdj = []
    adj = []
    verb = []
    noun = []
    adverb = []
    if language == "Latino":
        nounAdj = ['aria', 'arium', 'arius', 'atus', 'cola', 'colum', 'dicus', 'ellus', 'genus', 'gena', 'gen',
                   'mentum', 'or', 'tas', 'tus', 'ter', 'tio', 'tor', 'trix', 'trina', 'tudo', 'unculus', 'ura']
        adj = ['aceus', 'alis', 'andus', 'endus', 'iendus', 'ans', 'antis', 'ens', 'entis', 'iens', 'ientis', 'anus',
               'aticus', 'atus', 'bilis', 'bundus', 'ellus', 'ensis', 'esimus', 'eus', 'ilis', 'inus', 'ior', 'ius',
               'issimis', 'imus', 'osus', 'torius', 'timus', 'ulus']
        verb = ['esco', 'ico', 'ito', 'sco', 'so', 'sso', 'to', 'urio']

    if language == "Greco":
        noun = ['της', 'τής', 'ίτης', 'ώτης']
        adj = ['ῐος', 'εῖος']
        adverb = ['ως']
        verb = ['ίζω']

    for n in noun:
        if word.endswith(n):
            posWord[np.where(pos == "NOUN")] = 1
            return posWord

    for adv in adverb:
        if word.endswith(adv):
            posWord[np.where(pos == "ADV")] = 1
            return posWord

    for na in nounAdj:
        if word.endswith(na):
            posWord[np.where(pos == "NOUN")] = 0.5
            posWord[np.where(pos == "ADJ")] = 0.5
            return posWord

    for a in adj:
        if word.endswith(a):
            posWord[np.where(pos == "ADJ")] = 1
            return posWord

    for v in verb:
        if word.endswith(v):
            posWord[np.where(pos == "VERB")] = 1
            return posWord

    posWord[np.where(pos == "NOUN")] = 1
    return posWord
