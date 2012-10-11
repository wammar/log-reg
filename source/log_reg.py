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
from collections import defaultdict
from boilerpipe.extract import Extractor

class StoppingCriterion:
  # stop training when |loglikelihood(train data|new model) - loglikelihood(train data|prev model)| < threshold
  TRAIN_LIKELIHOOD=1 
  # stop training when loglikelihood(dev data|new model) - loglikelihood(dev data|prev model) < 0
  DEV_LIKELIHOOD=2
  
class OptimizationAlgorithm:
  STOCHASTIC_GRADIENT_DESCENT=1
  GRADIENT_DESCENT=2

class Regularizer:
  NONE=1
  L2=2

class LearningInfo:
  def __init__(self, 
               stoppingCriterion=StoppingCriterion.TRAIN_LIKELIHOOD, 
               stoppingCriterionThreshold = 0.00001, 
               positiveDevSetSize = 1, 
               negativeDevSetSize = 1,
               optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
               learningRate = 0.001,
               persistWeightsAtPrefix = '',
               regularizer = Regularizer.NONE,
               regularizationCoefficient = 0):
    self.stoppingCriterion = stoppingCriterion
    self.stoppingCriterionThreshold = stoppingCriterionThreshold
    self.positiveDevSetSize = positiveDevSetSize
    self.negativeDevSetSize = negativeDevSetSize
    self.optimizationAlgorithm = optimizationAlgorithm
    self.learningRate = learningRate
    self.persistWeightsAtPrefix = persistWeightsAtPrefix
    self.regularizer = regularizer
    self.regularizationCoefficient = regularizationCoefficient

  def REMOVE_ME(self):
    print 'DONOTHING'

# featuresDict is a defaultdict that maps non-zero feature ID to feature value in this example. label must be 0 or 1
class Example:
  def __init__(self, featuresDict, label):
    assert(label == 0 or label == 1)
    assert(type(featuresDict) == type(defaultdict(float)))
    self.featuresDict = featuresDict
    self.label = label
  
class LogisticRegression:

  # resets all feature weights to zeros
  def InitFeatureWeights(self):
    self.weights = defaultdict(float)

  def ReadExamples(self, labelFeaturesFilename):
    # read the examples in labelFeaturesFilename in a list 
    examples = []
    if labelFeaturesFilename != None and len(labelFeaturesFilename) > 0:
      labelFeaturesFile = open(labelFeaturesFilename)
      firstLine = True
      for exampleLine in labelFeaturesFile:
        # skip first line
        if firstLine:
          firstLine = False
          continue
        exampleLineSplits = exampleLine.strip().split()
        label = int(exampleLineSplits[0])
        featuresDict = defaultdict(float)
        for i in range(1, len(exampleLineSplits)):
          if i % 2 == 1:
            # feature ID
            featureId = exampleLineSplits[i]
          else:
            # feature value
            featureVal = float(exampleLineSplits[i])
            featuresDict[featureId] = featureVal
        newExample = Example(featuresDict, label)
        examples.append(newExample)
      labelFeaturesFile.close()

    return examples

  def __init__(self):

    # feature weights
    self.InitFeatureWeights()

  # fix exponent (before applying math.exp(exponent)) to avoid math range errors
  def FixExponent(self, exponent):
    if exponent > 600:
      exponent = 600
    return exponent

  # given an unlabeled example, compute probability that label=1, using the current feature weights
  def ComputeProb1(self, unlabeledExample):
    exponent = 0
#    print '===================\nunlabeledExample: \n{0}'.format(unlabeledExample)
    for featureId in unlabeledExample.featuresDict.keys():
      exponent -= unlabeledExample.featuresDict[featureId] * self.weights[featureId]
#      print 'computeprob1: {0}'.format(exponent)
    exponent = self.FixExponent(exponent)
    prob = 1.0 / (1.0 + math.exp(exponent))
    prob = self.fixProb(prob)
    return prob

  # slightly adjust probability to avoid math domain errors
  def fixProb(self, prob):
    if prob >= 1.0:
      prob -= 0.0001
    if prob <= 0.0:
      prob += 0.0001
    assert(prob < 1.0 and prob > 0.0)
    return prob

  # minimize the loss function using stochastic gradient descent
  # TODO: make sure L2 regularization works. and refactor the implementation such that other regularizers can be also used
  def MinimizeLossWithSGD(self, trainExamples, learningInfo, persistWeightsAt=''):

    # only TRAIN_LIKELIHOOD stopping criteria is implemented
    assert(learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LIKELIHOOD)

    # only L2 regularizer (and none) is implemented
    assert(learningInfo.regularizer == Regularizer.L2 or learningInfo.regularizer == Regularizer.NONE)

    converged = False
    prevIterLoss = 0

    # keep updating weights until the stopping criterion is satisfied
    iterationsCounter = 0
    while not converged:
      iterationsCounter += 1
      iterLoss = 0

      # for each training example
      for example in trainExamples:

        # compute p(class=1|example)
        prob1 = self.ComputeProb1(example)

        # update the iteration loss
        if example.label == 1:
          iterLoss -= math.log(prob1) / len(trainExamples)
        else:
          iterLoss -= math.log(1-prob1) / len(trainExamples)

        # update the weights of features firing in this example
        for featureId in example.featuresDict.keys():
          
          # compute the contribution of this example towards the l2-regularizer term of this feature 
          l2RegularizationUpdateTerm = learningInfo.regularizationCoefficient / len(trainExamples) * self.weights[featureId]

          # the actual weight update happens here
          self.weights[featureId] -= learningInfo.learningRate * ( example.featuresDict[featureId] * (prob1 - example.label) + l2RegularizationUpdateTerm )

        # end of features

      # end of training examples

      # now compute the feature vector's L2 norm
      l2RegularizationLossTerm = 0
      for featureId in self.weights.keys():
        l2RegularizationLossTerm += learningInfo.regularizationCoefficient / len(trainExamples) / 2.0 * self.weights[featureId] * self.weights[featureId]

      # add the L2 norm to the iteration loss
      iterLoss += l2RegularizationLossTerm

      # determine convergence
      if math.fabs(iterLoss - prevIterLoss) <= learningInfo.stoppingCriterionThreshold:
        print 'cost function converged after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, iterLoss, iterationsCounter)
        converged = True
      # update prevIterLoss for next iterations
      prevIterLoss = iterLoss

    # stopping criterion reached!

    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()

  # end of MinimizeLossWithSGD

  # minimize the loss function using batch gradient descent
  # TODO: add L2 regularization. and refactor the implementation such that other regularizers can be also used
  def MinimizeLossWithGD(self, trainExamples, learningInfo, persistWeightsAt=''):

    # only TRAIN_LIKELIHOOD stopping criteria is implemented
    assert(learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LIKELIHOOD)

    # no regularizer is implemented none
    assert(learningInfo.regularizer == Regularizer.NONE)

    converged = False
    prevIterLoss = 0

    # keep updating weights until stopping criterion
    iterationsCounter = 0
    gradient = defaultdict(float)
    while not converged:

      # new iteration, new gradient, new loss
      gradient.clear()
      iterationsCounter += 1
      iterLoss = 0

      # for each training example
      for example in trainExamples:

        # compute p(class=1|example) with current model params
        prob1 = self.ComputeProb1(example)

        # update iteration loss
        if example.label == 1:
          iterLoss -= math.log(prob1) / len(trainExamples)
        else:
          iterLoss -= math.log(1-prob1) / len(trainExamples)

        # for each feature that fire in this example
        for featureId in example.featuresDict.keys():

          # update the gradient
          gradient[featureId] += example.featuresDict[featureId] * (prob1 - example.label) 
        
        # end of features fired in this example

      # end of training examples

      # update the weights with the gradient
      for featureId in gradient.keys():
        self.weights[featureId] -= learningInfo.learningRate * (gradient[featureId] + learningInfo.regularizationCoefficient / len(trainExamples) * self.weights[featureId])

      # now compute the feature vector's L2 norm
      l2RegularizationLossTerm = 0
      for featureId in self.weights.keys():
        l2RegularizationLossTerm += learningInfo.regularizationCoefficient / len(trainExamples) / 2.0 * self.weights[featureId] * self.weights[featureId]
      iterLoss += l2RegularizationLossTerm

      print 'iterLoss = {0}'.format(iterLoss)

      # stopping criterion
      if math.fabs(iterLoss - prevIterLoss) <= learningInfo.stoppingCriterionThreshold:
        print 'cost function converged after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, iterLoss, iterationsCounter)
        converged = True
      prevIterLoss = iterLoss

    # end of training iterations

    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()

  # end of MinimizeLossWithGD
        
  # evaluate. returns (falsePositives, falseNegatives, truePositives, trueNegatives)
  def Evaluate(self, testExamples, classThreshold = 0.5):
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    for example in testExamples:
      prob1 = self.ComputeProb1(example)
      print 'prob1= {0} | label = {1}'.format(prob1, example.label)
      if prob1 >= classThreshold and example.label == 1:
        truePositives += 1
      elif prob1 < classThreshold and example.label == 0:
        trueNegatives += 1
      elif prob1 >= classThreshold and example.label == 0:
        falsePositives += 1
      else:
        falseNegatives += 1

    return (falsePositives, falseNegatives, truePositives, trueNegatives, prob1)

  # n-fold cross validation. returns (precision, recall, accuracy, f1)
  def NFoldCrossValidation(self, examples, n, learningInfo, classThreshold = 0.5):
    totalFalsePositives, totalFalseNegatives, totalTruePositives, totalTrueNegatives = 0, 0, 0, 0
    precision, recall, accuracy, f1 = 0, 0, 0, 0

    # only stochastic gradient descent and batch gradient descent are implemented
    assert(learningInfo.optimizationAlgorithm == OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT or 
           learningInfo.optimizationAlgorithm == OptimizationAlgorithm.GRADIENT_DESCENT)

    # find out which foldId each example belongs to
    examplesPerFold = int(math.ceil(len(examples) / float(n)))
    foldIds = []
    for i in range(0, len(examples)):
      foldIds.append(i/examplesPerFold)

    # iterate over the n folds
    for testFoldId in range(0, n):

      # identify train vs test examples
      trainExamples, testExamples = [], []
      for exampleId in range(0, len(examples)):
        if foldIds[exampleId] == testFoldId:
          testExamples.append(examples[exampleId])
        else:
          trainExamples.append(examples[exampleId])

      # optimize feature weights
      self.InitFeatureWeights()
      if len(learningInfo.persistWeightsAtPrefix) > 0:
        persistWeightsAt = '{0}.weights.{1}'.format(learningInfo.persistWeightsAtPrefix, testFoldId)
      else:
        persistWeightsAt = ''
      if learningInfo.optimizationAlgorithm == OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT:
        self.MinimizeLossWithSGD(trainExamples, learningInfo, persistWeightsAt)
      elif learningInfo.optimizationAlgorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
        self.MinimizeLossWithGD(trainExamples, learningInfo, persistWeightsAt)

      # evaluate model on test set
      (falsePositives, falseNegatives, truePositives, trueNegatives, prob1) = self.Evaluate(testExamples, classThreshold)
      # aggregate results
      totalFalsePositives += falsePositives
      totalFalseNegatives += falseNegatives
      totalTruePositives += truePositives
      totalTrueNegatives += trueNegatives

    # compute precision, recall, accuracy, and F1
    if totalTruePositives + totalFalsePositives > 0:
      precision = 1.0 * totalTruePositives / (totalTruePositives + totalFalsePositives)
    else:
      precision = 0
    if totalTruePositives + totalFalseNegatives > 0:
      recall = 1.0 * totalTruePositives / (totalTruePositives + totalFalseNegatives)
    else:
      recall = 0
    accuracy = 1.0 * (totalTruePositives + totalTrueNegatives) / (totalTruePositives + totalTrueNegatives + totalFalsePositives + totalFalseNegatives)
    if precision + recall > 0:
      f1 = 2.0 * precision * recall / (precision + recall)
    else:
      f1 = 0

    return (precision, recall, accuracy, f1)
