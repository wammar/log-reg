# coding: utf-8

import os
import shutil
import time
import sys
import io
import codecs
import re
import math
import json
import urllib2
import HTMLParser
import xml.dom.minidom
import socket
import operator
from collections import defaultdict
from boilerpipe.extract import Extractor

# controls
RUN_SUPERVISED = True
TUNE_HYPERPARAMS = False
REGULARIZE_SUPERVISED = True
DEFAULT_REGULARIZATION_COEFFICIENT = 0.01
DEFAULT_LEARNING_RATE = 0.01

RUN_SELF_LEARNING = False
REGULARIZE_UNSUPERVISED = True
SELF_LEARNING_ZERO_THRESHOLD = 0.01
SELF_LEARNING_ONE_THRESHOLD = 0.8

VERBOSE = True
USE_UNIGRAM_COUNTS = True
USE_BIGRAM_COUNTS = True
USE_TRIGRAM_COUNTS = True
USE_UNIGRAM_PRESENCE = False
USE_BIGRAM_PRESENCE = False
USE_TRIGRAM_PRESENCE = False
STOCHASTIC = True
FOLDS = 21

# parse command line arguments
pythonScript = sys.argv[0]
assert(pythonScript[-3:]=='.py')
rawDir = sys.argv[1]
outputPrefix = sys.argv[2]

#####################
# Utility Functions #
#####################

# compute and print metrics
def ComputeAndPrintMetrics(tp, tn, fp, fn, threshold):
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = 'N/A'
  if tp + fn > 0:
    recall = tp / (tp + fn)
  else:
    recall = 'N/A'
  if (tp + tn + fp + fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
  else:
    accuracy = 'N/A'
  if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
  else:
    f1 = 'N/A'
    
  # print threshold and metrics
  print '{0}\t{1}\t{2}\t{3}\t{4}'.format(threshold, precision, recall, accuracy, f1)

# tune the regularization coefficient
def TuneHyperparams(logReg, trainExamples, devExamples, learningInfo):
  coefficients = [0.01] #[0.1, 0.01, 0]
  learningRates = [0.001] #[0.001, 0.01, 0.1]

  # initialize the optimimum
  (bestCoefficient, bestLearningRate, bestDevLikelihood) = (0, 0, 0)

  # search a range of coefficients
  for coefficient in coefficients:

    # use the new coefficient
    learningInfo.regularizationCoefficient = coefficient
    print '\nNOW PLAYING WITH COEFFICIENT = {0}\n'.format(coefficient)

    # search a range of learning rates
    for learningRate in learningRates:

      # use the new learning rate
      learningInfo.learningRate = learningRate
      print '\nNOW PLAYING WITH LEARNING RATE = {0}\n'.format(learningRate)

      # do we need an internal dev set for the training procedure?
      if learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD:
        trainingTrainExamples = trainExamples[1:len(trainExamples)-1]
        trainingDevExamples = [trainExamples[0],trainExamples[-1]]
      elif learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD:
        trainingTrainExamples = trainExamples
        trainingDevExamples = []

      # train the model
      logReg.MinimizeLoss(
        trainExamples = trainingTrainExamples,
        devExamples = trainingDevExamples,
        learningInfo = learningInfo, 
        verbose = VERBOSE)

      # test the model on another dev set (different than the dev set used for training)
      (falsePositives, falseNegatives, truePositives, trueNegatives, exampleProb1Pairs) = logReg.Evaluate(testExamples = devExamples)

      # compute the likelihood of the latter dev set
      devLikelihood = 1.0
      for exampleProb1Pair in exampleProb1Pairs:
        if exampleProb1Pair[0].label == 1:
          devLikelihood *= exampleProb1Pair[1]
        else:
          devLikelihood *= 1-exampleProb1Pair[1]

      # did we find a better coefficient? (objective function is the likelihood of the latter dev set)
      if devLikelihood > bestDevLikelihood:
        print '\n###############UPDATE############'
        print 'best (coefficient, learning rate) was ({0}, {1})'.format(bestCoefficient, bestLearningRate)
        (bestCoefficient, bestLearningRate, bestDevLikelihood) = (coefficient, learningRate, devLikelihood)
        print 'best (coefficient, learning rate) became ({0}, {1})'.format(bestCoefficient, bestLearningRate)
        print '######END OF UPDATE##############\n'

    # end of learning rates

  # end of coefficients

  # return the best coefficient found
  return (bestCoefficient, bestLearningRate)

def SelfLearn(logReg, seedExamples, unlabeledExamples, testExample, learningInfo):
  
  # we will be messing with this list, we don't want to affect the original copy
  labeledExamples = list(seedExamples)
  
  # train the model with labeled examples
  logReg.MinimizeLoss(trainExamples=labeledExamples, devExamples=[], persistWeightsAt='', learningInfo=learningInfo, verbose = VERBOSE)

  # find unlabeled examples of which we are confident
  selfLearntExamples = []
  good1s = 0
  good0s = 0
  for unlabeledExampleId in range(0, len(unlabeledExamples)):
    prob1 = logReg.ComputeProb1(unlabeledExamples[unlabeledExampleId])
    if prob1 > SELF_LEARNING_ONE_THRESHOLD:
      good1s += 1
#      print 'adding {0}. prob1 = {1}. 1s = {2}'.format(unlabeledExampleId, prob1, good1s)
      newExample = unlabeledExamples[unlabeledExampleId]
      newExample.label = 1
      selfLearntExamples.append(newExample)
    elif prob1 < SELF_LEARNING_ZERO_THRESHOLD:
      good0s += 1
#      print 'adding {0}. prob1 = {1}. 0s = {2}'.format(unlabeledExampleId, prob1, good0s)
      newExample = unlabeledExamples[unlabeledExampleId]
      newExample.label = 0
      selfLearntExamples.append(newExample)
      
  # train the log linear model with self learnt examples
#  print '=============='

#  print '# original training examples = {0}'.format(len(labeledExamples))

#  print 'now, using {0} self learnt +ve examples and {1} self learnt -ve examples, totaling {2} out of {3} unlabeled examples'.format(good1s, good0s, len(selfLearntExamples), len(unlabeledExamples))

#  print 'Use self learnt examples, in addition to labeled ones'
  labeledExamples.extend(selfLearntExamples)

#  print '# total training examples = {0}'.format(len(labeledExamples))
  logReg.MinimizeLoss(labeledExamples, devExamples=[], persistWeightsAt=learningInfo.persistWeightsAtPrefix, learningInfo=learningInfo, verbose = VERBOSE)

  prob1 = logReg.ComputeProb1(testExample)
  print '****************************'
  print 'semi-supervised model score={0}\ttrue-labe={1}\t+ve-self-learnt-examples={2}\t-ve-self-learnt-examples={3}\t1-threshold={4}\t0-threshold={5}'.format(prob1, testExample.label, good1s, good0s, SELF_LEARNING_ONE_THRESHOLD, SELF_LEARNING_ZERO_THRESHOLD)
  learningInfo.Print()
  print '****************************'

  return prob1

def WriteFeaturesFile(docIdLabelFilename, featuresLabelFilename):
  # for each training example in raw/docId-label, write a line in features/labeled.txt. Instead of the site-ID in the original file, write the non-zero feature IDs and their values.
  docIdLabelFile = open(docIdLabelFilename)
  featuresLabelFile = open(featuresLabelFilename, mode='w')
  featuresLabelFile.write('lbl\tfeatId\tfeatVal\tfeatId\tfeatVal\t...\n')
  for docIdLabelLine in docIdLabelFile:
    # read name of the file from which features will be extracted, along with the label for this example
    (docId, label) = docIdLabelLine.strip().split()
    # write the label first
    featuresLabelFile.write('{0}'.format(label))
    # read the file
    doc = open('{0}/{1}'.format(rawDir, docId))
    docText = doc.read()
    doc.close()
    assert(len(docText) > 0)
    # extract features (unigrams, bigrams, trigrams)
    extractor.features.clear()
    extractor.BiasFeature('F0')
    if USE_UNIGRAM_COUNTS:
      extractor.ExtractNgramCount(docText, 1, 'F1')
    if USE_BIGRAM_COUNTS:
      extractor.ExtractNgramCount(docText, 2, 'F2')
    if USE_TRIGRAM_COUNTS:
      extractor.ExtractNgramCount(docText, 3, 'F3')
    if USE_UNIGRAM_PRESENCE:
      extractor.ExtractNgramPresence(docText, 1, 'F4')
    if USE_BIGRAM_PRESENCE:
      extractor.ExtractNgramPresence(docText, 2, 'F5')
    if USE_TRIGRAM_PRESENCE:
      extractor.ExtractNgramPresence(docText, 3, 'F6')
    # write the features in one line, in front of the label
    for key in extractor.features.keys():
      featuresLabelFile.write('\t{0}\t{1}'.format(key, extractor.features[key]))
    featuresLabelFile.write('\n')  
  # done with those files
  docIdLabelFile.close()
  featuresLabelFile.close()
  return

#################################
# importing in-progress modules #
#################################

featureExtractorModulePath = '/cab0/wammar/exp/feat-ext'
if featureExtractorModulePath not in sys.path:
  sys.path.append(featureExtractorModulePath)
from feature_extractor import FeatureExtractor
logResModulePath = '/cab0/wammar/exp/log-reg/source'
if logResModulePath not in sys.path:
  sys.path.append(logResModulePath)
from log_reg import *


#######################
# SUPERVISED LEARNING #
#######################

# initialize the feature extractor
extractor = FeatureExtractor('|')

# for each labeled example in raw/docId-label, write a line in features/labeled.txt. Instead of the site-ID in the original file, write the non-zero feature IDs and their values.
labeledFeaturesFilename = '{0}.labeled'.format(outputPrefix)
WriteFeaturesFile('{0}/docId-label.txt'.format(rawDir), labeledFeaturesFilename)

# initialize logistic regressin model
logReg = LogisticRegression()

# specify learning info
learningInfo = LearningInfo(
  stoppingCriterion = StoppingCriterion.TRAIN_LOGLIKELIHOOD,
  stoppingCriterionThreshold = 0.00001,
  positiveDevSetSize = 0, 
  negativeDevSetSize = 0,
  minTrainingIterationsCount = 3,
  persistWeightsAtPrefix = '{0}-supervisedWeights'.format(outputPrefix),
  maxTrainingIterationsCount = 500)

# stchastic?
if STOCHASTIC:
  learningInfo.optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
else:
  learningInfo.optimizationAlgorithm = OptimizationAlgorithm.GRADIENT_DESCENT

# load labeled examples
labeledExamples = logReg.ReadExamples(labeledFeaturesFilename)

# tune hyperparams
if TUNE_HYPERPARAMS:
  (bestRegularizationCoefficient, bestLearningRate) = TuneHyperparams(
    logReg = logReg, 
    trainExamples = labeledExamples[1:len(labeledExamples)-1], 
    devExamples = [labeledExamples[0], labeledExamples[-1]], 
    learningInfo = learningInfo)
  print 'best regularization coefficient = {0}'.format(bestRegularizationCoefficient)
  print 'best learning rate = {0}'.format(bestLearningRate)
  print '======================================\n\n'
else:
  (bestRegularizationCoefficient, bestLearningRate) = (DEFAULT_REGULARIZATION_COEFFICIENT, DEFAULT_LEARNING_RATE)

# learning rate?
learningInfo.learningRate = bestLearningRate

# regularizer?
if REGULARIZE_SUPERVISED:
  learningInfo.regularizationCoefficient = bestRegularizationCoefficient
  learningInfo.regularizer = Regularizer.L2
else:
  learningInfo.regularizationCoefficient = 0
  learningInfo.regularizer = Regularizer.NONE

# now, do LOO cross validation to give results
if RUN_SUPERVISED:
  (precision, recall, accuracy, f1, exampleProb1Pairs) = logReg.NFoldCrossValidation(
    examples=labeledExamples, 
    n=FOLDS, 
    learningInfo=learningInfo, 
    classThreshold=0.5, 
    verbose=VERBOSE,
    trackFeatureImpact=True)
  
  # find feature impacts
#  featureImpactPairs = []
#  for k,v in logReg.featureImpacts.iteritems():
#    featureImpactPairs.append((k, v))
  sortedFeatureImpactPairs = sorted(logReg.featureImpacts.iteritems(), key=operator.itemgetter(1))
  featureImpactFile = open('{0}-impact'.format(outputPrefix), 'w')
  for pair in sortedFeatureImpactPairs:
    featureImpactFile.write('{0}\t{1}\n'.format(pair[0], pair[1]))
  featureImpactFile.close()

  print '\nleave one out results:\n'
  print 'thrshld\tprec\trecall\tacc\tf1\n'
  decisionBoundaries = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
  for threshold in decisionBoundaries:
    # stats (true positives, true negatives, false positives, false negatives)
    (tp, tn, fp, fn) = (0.0, 0.0, 0.0, 0.0)
    for exampleProb1Pair in exampleProb1Pairs:
      if exampleProb1Pair[0].label == 1:
        if exampleProb1Pair[1] > threshold:
          tp += 1
        else:
          fn += 1
      else:
        if exampleProb1Pair[1] > threshold:
          fp += 1
        else:
          tn += 1
    # compute metrics
    ComputeAndPrintMetrics(tp = tp, tn = tn, fp = fp, fn = fn, threshold = threshold)

#################
# SELF LEARNING #
#################

if REGULARIZE_UNSUPERVISED:
  learningInfo.regularizationCoefficient = bestRegularizationCoefficient
  learningInfo.regularizer = Regularizer.L2
else:
  learningInfo.regularizationCoefficient = 0
  learningInfo.regularizer = Regularizer.NONE

if RUN_SELF_LEARNING:
  # now train one good system to generate labels for unlabeled documents
  logReg.MinimizeLoss(
    trainExamples = labeledExamples,
    learningInfo = learningInfo,
    verbose = VERBOSE)

  # for each unlabeled example in raw/docId-randomLabel, write a line in features/randomLabeled.txt.
  unlabeledFeaturesFilename = '{0}.unlabeled'.format(outputPrefix)
  if not os.path.exists(unlabeledFeaturesFilename):
    WriteFeaturesFile('{0}/docId-randomLabel.txt'.format(rawDir), unlabeledFeaturesFilename)

  # then, read unlabeledExamples
  unlabeledExamples = logReg.ReadExamples(unlabeledFeaturesFilename)

  # leave-one-out with self learning
  learningInfo.persistWeightsAtPrefix = '{0}-unsupervisedWeights'.format(outputPrefix)
  resultPairs = []
  for i in range(0, len(labeledExamples)):
    print 'now using {0} as a test example'.format(i)
    testExample = labeledExamples[i]
    seedExamples = labeledExamples[0:i] + labeledExamples[i+1:]
    prob1 = SelfLearn(logReg, seedExamples, unlabeledExamples, testExample, learningInfo=learningInfo)
    # remember what happened
    resultPairs.append( (testExample, prob1) )

  # compute metrics
  print '\nleave one out results:\n'
  print 'thrshld\tprec\trecall\tacc\tf1\n'
  decisionBoundaries = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
  for threshold in decisionBoundaries:
    for pair in resultPairs:
      # remember what happened
      if pair[0].label == 1:
        if pair[1] > threshold:
          tp += 1
        else:
          fn += 1
      else:
        if pair[1] > threshold:
          fp += 1
        else:
          tn += 1
    ComputeAndPrintMetrics(fp = fp, tp = tp, fn = fn, tn = tn, threshold = threshold)
