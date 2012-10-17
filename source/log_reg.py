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

class Const:
  MAX_EXPONENT = 300.0
  MAX_NUMBER = math.exp(300.0)

class StoppingCriterion:
  # stop training when |loglikelihood(train data|new model) - loglikelihood(train data|prev model)| < threshold
  TRAIN_LOGLIKELIHOOD=1 
  # stop training when unregularized-log-likelihood(dev data|new model) - unregularized-log-likelihood(dev data|prev model) < 0
  DEV_LOGLIKELIHOOD=2
  
class OptimizationAlgorithm:
  STOCHASTIC_GRADIENT_DESCENT=1
  GRADIENT_DESCENT=2

class Regularizer:
  NONE=1
  L2=2

class LearningInfo:
  def __init__(self, 
               stoppingCriterion=StoppingCriterion.TRAIN_LOGLIKELIHOOD, 
               stoppingCriterionThreshold = 0.00001, 
               positiveDevSetSize = 1,
               negativeDevSetSize = 1,
               optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
               learningRate = 0.001,
               persistWeightsAtPrefix = '',
               regularizer = Regularizer.NONE,
               regularizationCoefficient = 0,
               minTrainingIterationsCount = 0,
               maxTrainingIterationsCount = 1000):
    self.stoppingCriterion = stoppingCriterion
    self.stoppingCriterionThreshold = stoppingCriterionThreshold
    self.positiveDevSetSize = positiveDevSetSize
    self.negativeDevSetSize = negativeDevSetSize
    self.optimizationAlgorithm = optimizationAlgorithm
    self.learningRate = learningRate
    self.persistWeightsAtPrefix = persistWeightsAtPrefix
    self.regularizer = regularizer
    self.regularizationCoefficient = regularizationCoefficient
    self.minTrainingIterationsCount = minTrainingIterationsCount
    self.maxTrainingIterationsCount = maxTrainingIterationsCount

  def Print(self):
    print 'stoppingCriterion = {0}'.format(self.stoppingCriterion)
    print 'stoppingCriterionThreshold = {0}'.format(self.stoppingCriterionThreshold)
    print 'positiveDevSetSize = {0}'.format(self.positiveDevSetSize)
    print 'negativeDevSetSize = {0}'.format(self.negativeDevSetSize)
    print 'optimizationAlgorithm = {0}'.format(self.optimizationAlgorithm)
    print 'learningRate = {0}'.format(self.learningRate)
    print 'persistWeightsAtPrefix = {0}'.format(self.persistWeightsAtPrefix)
    print 'regularizer = {0}'.format(self.regularizer)
    print 'regularizationCoefficient = {0}'.format(self.regularizationCoefficient)
    print 'minTrainingIterationsCount = {0}'.format(self.minTrainingIterationsCount)
    print 'maxTrainingIterationsCount = {0}'.format(self.maxTrainingIterationsCount)

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
    self.featureImpacts = defaultdict(float)

  def ReadExamples(self, labelFeaturesFilename):
    # read the examples in labelFeaturesFilename in a list 
    examples = []
    if labelFeaturesFilename != None and len(labelFeaturesFilename) > 0:
      print 'labelFeaturesFilename={0}'.format(labelFeaturesFilename)
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
    if exponent > Const.MAX_EXPONENT:
      exponent = Const.MAX_EXPONENT
    return exponent

  # given an unlabeled example, compute probability that label=1, using the current feature weights
  def ComputeProb1(self, unlabeledExample, trackFeatureImpact=False):
    exponent = 0
#    print '===================\nunlabeledExample: \n{0}'.format(unlabeledExample)
    for featureId in unlabeledExample.featuresDict.keys():
      temp = unlabeledExample.featuresDict[featureId] * self.weights[featureId]
      if math.isnan(temp):
        continue
      if trackFeatureImpact:
        self.featureImpacts[featureId] += math.fabs(temp)
      exponent -= temp
      # TODO:  better handling of this error
      if math.isnan(exponent):
        exponent = 0
    exponent = self.FixExponent(exponent)
    prob = 1.0 / (1.0 + math.exp(exponent))
    prob = self.fixProb(prob, exponent)
    return prob

  # slightly adjust probability to avoid math domain errors
  def fixProb(self, prob, exponent='-1'):
    if prob >= 1.0:
      prob -= 0.0001
    if prob <= 0.0:
      prob += 0.0001

    x = False
    if prob <= 1.0 and prob >= 0.0:
      x = True
    else:
      print 'problematic prob = {0}\noriginal exponent={1}'.format(prob, exponent)

    assert(x)
    return prob

  # am i a huge positive or negative number?
  def AmIBig(self, x):
    return math.fabs(x) > Const.MAX_NUMBER

  # determines whether the model has converged, based on the stopping criterion specified in learningInfo
  def IsModelConverged(self, learningInfo, iterNumber, prevIterLoss=0, thisIterLoss=0, prevDevLoglikelihood=0, thisDevLoglikelihood=0, verbose=False):

    # bounds on the number of iterations
    if iterNumber < learningInfo.minTrainingIterationsCount:
      return False
    if iterNumber >  learningInfo.maxTrainingIterationsCount:
      if verbose:
        print 'forced to converge after reaching the maximum allowed number of iterations {0}'.format(learningInfo.maxTrainingIterationsCount)
      return True

    # stop when dev loglikelihood stabilizes
    if learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD:
      
      # check the difference
      if thisDevLoglikelihood - prevDevLoglikelihood <= learningInfo.stoppingCriterionThreshold:
        if verbose:
          print 'cost function converged after {2} iterations: prevDevLogLikelihood = {0} | thisDevLogLikelihood = {1}'.format(
            prevDevLoglikelihood, 
            thisDevLoglikelihood,
            iterNumber)
        return True
      else:
        return False
    # end of dev_loglikelihood stopping criteria

    # stop when train loglikelihood stabiliizes
    elif learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD:

      # if the loss doubles, something is wrong
      if prevIterLoss > 0 and thisIterLoss / prevIterLoss > 2.0 and iterNumber > 10:
        if verbose:
          print 'forced to converge cuz the iteration loss doubled since the previous iteration, after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, thisIterLoss, iterNumber)
        return True

      # check the difference
      if math.fabs(thisIterLoss - prevIterLoss) <= learningInfo.stoppingCriterionThreshold:
        if verbose:
          print 'cost function converged after {2} iterations: prevIterLoss = {0} | iterLoss = {1}'.format(prevIterLoss, thisIterLoss, iterNumber)
        return True
      else:
        return False
    # end of dev_loglikelihood stopping criteria

  # mere wrapper
  def MinimizeLoss(self, trainExamples, learningInfo, devExamples=[], persistWeightsAt='', resetWeights=True, verbose=False):
    if learningInfo.optimizationAlgorithm == OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT:
      self.MinimizeLossWithSGD(trainExamples=trainExamples, learningInfo=learningInfo, 
                          devExamples=devExamples, persistWeightsAt=persistWeightsAt, resetWeights=resetWeights, verbose=verbose)
    elif learningInfo.optimizationAlgorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
      self.MinimizeLossWithGD(trainExamples=trainExamples, learningInfo=learningInfo, 
                          devExamples=devExamples, persistWeightsAt=persistWeightsAt, resetWeights=resetWeights, verbose=verbose)

  # minimize the loss function using stochastic gradient descent
  # TODO:  refactor the l2 implementation such that other regularizers can be also used
  def MinimizeLossWithSGD(self, trainExamples, learningInfo, devExamples=[], persistWeightsAt='', resetWeights=True, verbose=False):

    # it's a good idea to do this by default to avoid accidental influence of one experiment on the other
    if resetWeights:
      self.InitFeatureWeights()

    # only TRAIN_LOGLIKELIHOOD stopping criteria is implemented
    assert(learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD or
           learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD and len(devExamples) > 0)

    # only L2 regularizer (and none) is implemented
    assert(learningInfo.regularizer == Regularizer.L2 or learningInfo.regularizer == Regularizer.NONE)

    # keep updating weights until the stopping criterion is satisfied
    converged = False
    prevIterLoss = 0
    prevDevLoglikelihood = float('-inf')
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
      if verbose and learningInfo.regularizer != Regularizer.NONE:
        print 'regularizationLossTerm = {0}'.format(l2RegularizationLossTerm)

      # add the L2 norm to the iteration loss
      iterLoss += l2RegularizationLossTerm

      # compute the unregularized log likelihood of dev set
      devLoglikelihood = 0
      for example in devExamples:
        prob1 = self.ComputeProb1(example)
        if example.label == 1:
          devLoglikelihood += math.log(prob1) / len(trainExamples)
        else:
          devLoglikelihood += math.log(1-prob1) / len(trainExamples)

      # a concise message summarizing the progress made in this iteration
      if learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD and verbose:
        print 'devLoglikelihood = {0}'.format(devLoglikelihood)
      elif learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD and verbose:
        print 'iterLoss = {0}'.format(iterLoss)

      # determine convergence
      if self.IsModelConverged(iterNumber=iterationsCounter, thisIterLoss=iterLoss, 
                               prevIterLoss=prevIterLoss, learningInfo=learningInfo, 
                               prevDevLoglikelihood=prevDevLoglikelihood, thisDevLoglikelihood=devLoglikelihood, verbose=verbose):
        converged = True
      
      # update prevIterLoss for next iterations
      prevIterLoss = iterLoss
      prevDevLoglikelihood = devLoglikelihood

    # end of training iterations -- stopping criterion reached!

    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()

  # end of MinimizeLossWithSGD

  # minimize the loss function using batch gradient descent
  def MinimizeLossWithGD(self, trainExamples, learningInfo, devExamples=[], persistWeightsAt='', resetWeights=True, verbose=False):

    # it's a good idea to do this by default to avoid accidental influence of one experiment on the other
    if resetWeights:
      self.InitFeatureWeights()

    # only TRAIN_LOGLIKELIHOOD stopping criteria is implemented
    assert(learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD or
           learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD and len(devExamples) > 0)

    # no regularizer is implemented none
    assert(learningInfo.regularizer == Regularizer.NONE or learningInfo.regularizer == Regularizer.L2)

    # keep updating weights until stopping criterion
    iterationsCounter = 0
    gradient = defaultdict(float)
    converged = False
    prevIterLoss = 0
    prevDevLoglikelihood = float('-inf')
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
        
        # loss is too big already! decision: consider this model converged.
        if iterLoss == float('inf') or iterLoss == float('nan'):
          exponent = 400
        elif iterLoss == 0.0:
          exponent = 1
        else:
          exponent = math.log(iterLoss)
        if self.FixExponent(exponent) != exponent:
          if verbose:
            print '*&*(^%%$#*()*@$(*(*&&*!#$_)*$)('
            print ' iteration loss sky rocketed!'
            print '*&*(^%%$#*()*@$(*(*&&*!#$_)*$)('
          converged = True
          break

        # for each feature that fire in this example
        for featureId in example.featuresDict.keys():

          # update the gradient
          gradient[featureId] += example.featuresDict[featureId] * (prob1 - example.label) 
        
        # end of features fired in this example

      # end of training examples

      # update the weights with the gradient
      for featureId in gradient.keys():
        
        # regularizer contribution to the update
        regularizerUpdate = 0
        if learningInfo.regularizer == Regularizer.L2:
          regularizerUpdate = learningInfo.regularizationCoefficient / len(trainExamples) * self.weights[featureId]

        # the actual weight update
        self.weights[featureId] -= learningInfo.learningRate * (gradient[featureId] + regularizerUpdate)

      # now compute the regularization term in the objective function
      regularizationLossTerm = 0
      for featureId in self.weights.keys():
        if learningInfo.regularizer == Regularizer.L2:
          regularizationLossTerm += learningInfo.regularizationCoefficient / len(trainExamples) / 2.0 * self.weights[featureId] * self.weights[featureId]

      # add the regularization term
      if verbose and learningInfo.regularizer != Regularizer.NONE:
        print 'regularizationLossTerm = {0}'.format(regularizationLossTerm)
      iterLoss += regularizationLossTerm

      # compute the unregularized log likelihood of dev set
      devLoglikelihood = 0
      for example in devExamples:
        prob1 = self.ComputeProb1(example)
        if example.label == 1:
          devLoglikelihood += math.log(prob1) / len(trainExamples)
        else:
          devLoglikelihood += math.log(1-prob1) / len(trainExamples)

      # a concise message summarizing the progress made in this iteration
      if learningInfo.stoppingCriterion == StoppingCriterion.DEV_LOGLIKELIHOOD and verbose:
        print 'devLoglikelihood = {0}'.format(devLoglikelihood)
      elif learningInfo.stoppingCriterion == StoppingCriterion.TRAIN_LOGLIKELIHOOD and verbose:
        print 'iterLoss = {0}'.format(iterLoss)

      # determine convergence
      if self.IsModelConverged(iterNumber=iterationsCounter, thisIterLoss=iterLoss, 
                               prevIterLoss=prevIterLoss, learningInfo=learningInfo, 
                               prevDevLoglikelihood=prevDevLoglikelihood, thisDevLoglikelihood=devLoglikelihood, verbose=verbose):
        converged = True
      
      # update prevIterLoss for next iterations
      prevIterLoss = iterLoss
      prevDevLoglikelihood = devLoglikelihood

    # end of training iterations -- stopping criterion reached

    # persist optimal weights
    if len(persistWeightsAt) > 0:
      weightsFile = open(persistWeightsAt, 'w')
      for key in self.weights.keys():
        weightsFile.write('{0}\t{1}\n'.format(key, self.weights[key]))
      weightsFile.close()

  # end of MinimizeLossWithGD
        
  # evaluate. returns (falsePositives, falseNegatives, truePositives, trueNegatives, prob1)
  def Evaluate(self, testExamples, classThreshold = 0.5, verbose=True, trackFeatureImpact=False):
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    exampleProb1List = []
    for example in testExamples:
      prob1 = self.ComputeProb1(example, trackFeatureImpact)
      exampleProb1List.append((example, prob1))
      print 'prob1= {0} | label = {1}'.format(prob1, example.label)
      if prob1 >= classThreshold and example.label == 1:
        truePositives += 1
      elif prob1 < classThreshold and example.label == 0:
        trueNegatives += 1
      elif prob1 >= classThreshold and example.label == 0:
        falsePositives += 1
      else:
        falseNegatives += 1

    return (falsePositives, falseNegatives, truePositives, trueNegatives, exampleProb1List)

  # n-fold cross validation. returns (precision, recall, accuracy, f1)
  def NFoldCrossValidation(self, examples, n, learningInfo, classThreshold = 0.5, verbose=False, trackFeatureImpact=False):

    if verbose:
      learningInfo.Print()

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

    # example-prob pairs
    exampleProb1Pairs = []

    # iterate over the n folds
    for testFoldId in range(0, n):

      # identify train, dev and test examples for this fold
      trainExamples, testExamples, positiveDevExamples, negativeDevExamples = [], [], [], []
      for exampleId in range(0, len(examples)):
        if foldIds[exampleId] == testFoldId:
          testExamples.append(examples[exampleId])
        elif len(positiveDevExamples) < learningInfo.positiveDevSetSize and examples[exampleId].label == 1:
          positiveDevExamples.append(examples[exampleId])
        elif len(negativeDevExamples) < learningInfo.negativeDevSetSize and examples[exampleId].label == 0:
          negativeDevExamples.append(examples[exampleId])
        else:
          trainExamples.append(examples[exampleId])
      # end of examples

      # merge positive and negative dev examples
      devExamples = negativeDevExamples + positiveDevExamples
      
      # optimize feature weights
      self.InitFeatureWeights()
      if len(learningInfo.persistWeightsAtPrefix) > 0:
        persistWeightsAt = '{0}.weights.{1}'.format(learningInfo.persistWeightsAtPrefix, testFoldId)
      else:
        persistWeightsAt = ''
      if learningInfo.optimizationAlgorithm == OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT:
        self.MinimizeLossWithSGD(trainExamples=trainExamples, devExamples=devExamples, 
                                 learningInfo=learningInfo, persistWeightsAt=persistWeightsAt, verbose=verbose)
      elif learningInfo.optimizationAlgorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
        self.MinimizeLossWithGD(trainExamples=trainExamples, devExamples=devExamples, 
                                learningInfo=learningInfo, persistWeightsAt=persistWeightsAt, verbose=verbose)

      # evaluate model on test set
      (falsePositives, falseNegatives, truePositives, trueNegatives, foldExampleProb1Pairs) = self.Evaluate(testExamples, classThreshold, verbose=verbose, trackFeatureImpact=trackFeatureImpact)

      # update the exampleProbPairs
      exampleProb1Pairs += foldExampleProb1Pairs

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

    return (precision, recall, accuracy, f1, exampleProb1Pairs)
