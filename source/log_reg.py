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

  def __init__(self, labelFeaturesFilename, learningRate, maxIterLossDiff, regularizationConst = 0):
    # regularization const
    self.regularizationConst = regularizationConst

    # learning rate
    self.learningRate = learningRate
    assert(self.learningRate > 0)

    # convergence criterion
    self.maxIterLossDiff = maxIterLossDiff
    assert(self.maxIterLossDiff > 0)
    
    # feature weights
    self.InitFeatureWeights()

    # read examples in labelFeaturesFilename in a list self.examples
    self.examples = self.ReadExamples(labelFeaturesFilename)
            
  # fix exponent (before applying math.exp(exponent)) to avoid math range errors
  def FixExponent(self, exponent):
    if exponent > 600:
      exponent = 600
    return exponent

  # given an unlabeled example, compute probability that label=1, using the current feature weights
  def ComputeProb1(self, unlabeledExample):
    exponent = 0
#    print '===================computeprob1Init: {0}'.format(exponent)
    for featureId in unlabeledExample.featuresDict.keys():
      exponent -= unlabeledExample.featuresDict[featureId] * self.weights[featureId]
#      print 'computeprob1: {0}'.format(exponent)
    exponent = self.FixExponent(exponent)
    prob = 1.0 / (1.0 + math.exp(exponent))
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
  def MinimizeLossWithSGD(self, trainExamples, persistWeightsAt=''):
    converged = False
    prevIterLoss = 0
    # keep updating weights until the total iteration loss stabilizes
    iterationsCounter = 0
    while not converged:
      iterationsCounter += 1
      iterLoss = 0
      for example in trainExamples:
        prob1 = self.ComputeProb1(example)
        prob1 = self.fixProb(prob1)
        if example.label == 1:
          iterLoss -= math.log(prob1) / len(trainExamples)
        else:
          iterLoss -= math.log(1-prob1) / len(trainExamples)
        for featureId in example.featuresDict.keys():
          # update the weights
          l2RegularizationUpdateTerm = self.regularizationConst / len(trainExamples) * self.weights[featureId]
          self.weights[featureId] -= self.learningRate * ( example.featuresDict[featureId] * (prob1 - example.label) + l2RegularizationUpdateTerm )
      # now compute the feature vector's L2 norm
      l2RegularizationLossTerm = 0
      for featureId in self.weights.keys():
        l2RegularizationLossTerm += self.regularizationConst / len(trainExamples) / 2.0 * self.weights[featureId] * self.weights[featureId]
      iterLoss += self.regularizationConst * l2RegularizationLossTerm
#      print 'iterLoss = {0}'.format(iterLoss)
      if math.fabs(iterLoss - prevIterLoss) <= self.maxIterLossDiff:
        print 'cost function converged after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, iterLoss, iterationsCounter)
        converged = True
      prevIterLoss = iterLoss
    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()
        
  # minimize the loss function using batch gradient descent
  # TODO: add L2 regularization works. and refactor the implementation such that other regularizers can be also used
  def MinimizeLossWithGD(self, trainExamples, persistWeightsAt=''):
    converged = False
    prevIterLoss = 0
    # keep updating weights until the total iteration loss stabilizes
    iterationsCounter = 0
    gradient = defaultdict[float]
    while not converged:
      gradient.clear()
      iterationsCounter += 1
      iterLoss = 0
      for example in trainExamples:
        prob1 = self.ComputeProb1(example)
        prob1 = self.fixProb(prob1)
        if example.label == 1:
          iterLoss -= math.log(prob1) / len(trainExamples)
        else:
          iterLoss -= math.log(1-prob1) / len(trainExamples)
        for featureId in example.featuresDict.keys():
          # update the gradient
          gradient[featureId] += example.featuresDict[featureId] * (prob1 - example.label) 

      # update the weights
      for featureId in gradient.keys():
        self.weights[featureId] -= self.learningRate * gradient[featureId]

      # now compute the feature vector's L2 norm
#      l2RegularizationLossTerm = 0
#      for featureId in self.weights.keys():
#        l2RegularizationLossTerm += self.regularizationConst / len(trainExamples) / 2.0 * self.weights[featureId] * self.weights[featureId]
#      iterLoss += self.regularizationConst

#      print 'iterLoss = {0}'.format(iterLoss)
      if math.fabs(iterLoss - prevIterLoss) <= self.maxIterLossDiff:
        print 'cost function converged after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, iterLoss, iterationsCounter)
        converged = True
      prevIterLoss = iterLoss
    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()
        
  # evaluate. returns (falsePositives, falseNegatives, truePositives, trueNegatives)
  def Evaluate(self, testExamples):
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    for example in testExamples:
      prob1 = self.ComputeProb1(example)
      prob1 = self.fixProb(prob1)
      print 'prob1= {0} | label = {1}'.format(prob1, example.label)
      if prob1 >= 0.5 and example.label == 1:
        truePositives += 1
      elif prob1 < 0.5 and example.label == 0:
        trueNegatives += 1
      elif prob1 >= 0.5 and example.label == 0:
        falsePositives += 1
      else:
        falseNegatives += 1

    return (falsePositives, falseNegatives, truePositives, trueNegatives, prob1)

  # n-fold cross validation. returns (precision, recall, accuracy, f1)
  def NFoldCrossValidation(self, n, persistWeightsAtPrefix = '', optimizationAlgorithm = 'SGD'):
    totalFalsePositives, totalFalseNegatives, totalTruePositives, totalTrueNegatives = 0, 0, 0, 0
    precision, recall, accuracy, f1 = 0, 0, 0, 0

    # only stochastic gradient descent and batch gradient descent are implemented
    assert(optimizationAlgorithm == 'SGD' or optimizationAlgorith == 'GD')

    # find out which foldId each example belongs to
    examplesPerFold = int(math.ceil(len(self.examples) / float(n)))
    foldIds = []
    for i in range(0, len(self.examples)):
      foldIds.append(i/examplesPerFold)

    # iterate over the n folds
    for testFoldId in range(0, n):
      # identify train vs test examples
      trainExamples, testExamples = [], []
      for exampleId in range(0, len(self.examples)):
        if foldIds[exampleId] == testFoldId:
          testExamples.append(self.examples[exampleId])
        else:
          trainExamples.append(self.examples[exampleId])
      # optimize feature weights on train set
      self.InitFeatureWeights()
      if len(persistWeightsAtPrefix) > 0:
        persistWeightsAt = '{0}.weights.{1}'.format(persistWeightsAtPrefix, testFoldId)
      else:
        persistWeightsAt = ''
      if optimizationAlgorithm == 'SGD':
        self.MinimizeLossWithSGD(trainExamples, persistWeightsAt)
      elif optimizationAlgorithm == 'GD':
        self.MinimizeLossWithGD(trainExamples, persistWeightsAt)
      # evaluate model on test set
      (falsePositives, falseNegatives, truePositives, trueNegatives) = self.Evaluate(testExamples)
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
