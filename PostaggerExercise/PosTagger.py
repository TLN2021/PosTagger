# -*- coding: utf-8 -*-
import Evaluation as ev
import Utils as utils
import LearningPhase as lp
import Smoothing as sm
import DecodingPhase as dp

def main(language):
    if language == "Latino":
        trainSetFile = '../TreeBank - Latino/la_llct-ud-train.conllu'
        testSetFile = '../TreeBank - Latino/la_llct-ud-test.conllu'
        devSetFile = '../TreeBank - Latino/la_llct-ud-dev.conllu'
    elif language == "Greco":
        trainSetFile = '../TreeBank - Greco/grc_perseus-ud-train.conllu'
        testSetFile = '../TreeBank - Greco/grc_perseus-ud-test.conllu'
        devSetFile = '../TreeBank - Greco/grc_perseus-ud-dev.conllu'

    sentenceTest, correctPos = ev.getSencencePos(testSetFile)

    # 1) LEARNING (sul training set)
    print('Start Learning..')
    with open(trainSetFile, 'r', encoding='utf-8') as trainFile:
        posInTrain = utils.findingAllPos(trainFile)
        transitionProbabilityMatrix, emissionProbabilityDictionary = lp.learningPhase(trainFile, posInTrain)

    # 1.5) SMOOTHING
    # Sussistono 4 modalità di smoothing + 1 (vd riga 42)
    # smoothingType = 0: unknown -> NOUN
    # smoothingType = 1: unknown -> NOUND/VERB
    # smoothingType = 2: unknown -> distribution on pos
    # smoothingType = 3: unknown -> distribution on dev set
    smoothingType = 0
    smoothingVector = sm.smoothing(posInTrain, smoothingType, devSetFile)

    # 2) DECODING (sul test set)
    print('Start Decoding..')
    viterbiPos = []
    for sentence in sentenceTest:
        #viterbiPos.append(dp.viterbiAlgorithm(sentence, posInTrain, transitionProbabilityMatrix, emissionProbabilityDictionary, smoothingVector))
        
        # smoothingType implementato
        viterbiPos.append(dp.syntaxBasedDecoding(sentence, posInTrain, transitionProbabilityMatrix, emissionProbabilityDictionary, language))

    accuracyOnTest = ev.accuracy(correctPos, viterbiPos)
    print('Accuracy con smoothing di tipo ',smoothingType,'sul ',language,' : ', accuracyOnTest)


#-------------------------------------------------------

language = "Latino"
#language = "Greco"
main(language)



