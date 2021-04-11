# -*- coding: utf-8 -*-
import Evaluation as ev
import Utils as utils
import LearningPhase as lp
import Smoothing as sm
import DecodingPhase as dp

def main(language):
    if language == "Latino":
        train_set_file = 'TreeBank - Latino/la_llct-ud-train.conllu'
        testSetFile = 'TreeBank - Latino/la_llct-ud-test.conllu'
        devSetFile = 'TreeBank - Latino/la_llct-ud-dev.conllu'
    elif language == "Greco":
        train_set_file = 'TreeBank - Greco/grc_perseus-ud-train.conllu'
        #testSetFile = 'TreeBank - Greco/grc_perseus-ud-test.conllu'
        testSetFile = 'TreeBank - Greco/prova.txt'
        devSetFile = 'TreeBank - Greco/grc_perseus-ud-dev.conllu'

    sentenceTest, correctPos = ev.getSencencePos(testSetFile)
    print("numero parole", len(sentenceTest))
    # 1) LEARNING (sul training set)
    with open(train_set_file, 'r', encoding='utf-8') as trainFile:
        posInTrain = utils.findingAllPos(trainFile)
        transitionProbabilityMatrix, emissionProbabilityDictionary = lp.learningPhase(trainFile, posInTrain)

    # 1.5) SMOOTHING
    smoothingType = 1
    smoothingVector = sm.smoothing(posInTrain, smoothingType, devSetFile)

    # 2) DECODING (sul test set)
    count=0
    viterbiPos = []
    for sentence in sentenceTest:
        count+=1
        viterbiPos.append(dp.viterbiAlgorithm(sentence, posInTrain, transitionProbabilityMatrix, emissionProbabilityDictionary, smoothingVector))
        #viterbiPos.append(dp.syntaxBasedDecoding(sentence, posInTrain, transitionProbabilityMatrix, emissionProbabilityDictionary, language))
    print("numero di iterazioni",count)
    accuracyOnTest = ev.accuracy(correctPos, viterbiPos)
    print(accuracyOnTest)


#-------------------------------------------------------

#language = "Latino"
language = "Greco"
main(language)



