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

def SelfLearn(logReg, labeledExamples, unlabeledExamples, testExample):
  # train the model with labeled examples
  logReg.MinimizeLossWithSGD(labeledExamples, persistWeightsAt='')

  # find unlabeled examples of which we are confident
  selfLearntExamples = []
  good1s = 0
  good0s = 0
  for unlabeledExampleId in range(0, len(unlabeledExamples)):
    prob1 = logReg.ComputeProb1(unlabeledExamples[unlabeledExampleId])
    if prob1 > 0.8:
      good1s += 1
#      print 'adding {0}. prob1 = {1}. 1s = {2}'.format(unlabeledExampleId, prob1, good1s)
      newExample = unlabeledExamples[unlabeledExampleId]
      newExample.label = 1
      selfLearntExamples.append(newExample)
    elif prob1 < 0.01:
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
  logReg.MinimizeLossWithSGD(labeledExamples, persistWeightsAt='')

  prob1 = logReg.ComputeProb1(testExample)
#  print '****************************'
  print '{0}\t{1}'.format(prob1, testExample.label)
#  print '****************************'

  return

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
    extractor.ExtractNgramCount(docText, 1, 'F1')
    extractor.ExtractNgramCount(docText, 2, 'F2')
    extractor.ExtractNgramCount(docText, 3, 'F3')
    extractor.BiasFeature('F4')
    #  extractor.ExtractNgramPresence(docText, 1, 'F4')
    #  extractor.ExtractNgramPresence(docText, 2, 'F5')
    #  extractor.ExtractNgramPresence(docText, 3, 'F6')
    # write the features in one line, in front of the label
    for key in extractor.features.keys():
      featuresLabelFile.write('\t{0}\t{1}'.format(key, extractor.features[key]))
    featuresLabelFile.write('\n')  
  # done with those files
  docIdLabelFile.close()
  featuresLabelFile.close()
  return

# parse command line arguments
pythonScript = sys.argv[0]
assert(pythonScript[-3:]=='.py')
rawDir = sys.argv[1]
outputPrefix = sys.argv[2]

# importing in-progress modules
featureExtractorModulePath = '/cab0/wammar/exp/feat-ext'
if featureExtractorModulePath not in sys.path:
  sys.path.append(featureExtractorModulePath)
from feature_extractor import FeatureExtractor
logResModulePath = '/cab0/wammar/exp/log-reg/source'
if logResModulePath not in sys.path:
  sys.path.append(logResModulePath)
from log_reg import *

# initialize the feature extractor
extractor = FeatureExtractor('|')

# for each labeled example in raw/docId-label, write a line in features/labeled.txt. Instead of the site-ID in the original file, write the non-zero feature IDs and their values.
labeledFeaturesFilename = '{0}.labeled'.format(outputPrefix)
WriteFeaturesFile('{0}/docId-label.txt'.format(rawDir), labeledFeaturesFilename)

# initialize logistic regressin model
logReg = LogisticRegression()

# specify learning info
learningInfo = LearningInfo(
  learningRate = 0.001, 
  stoppingCriterion = StoppingCriterion.TRAIN_LIKELIHOOD,
  stoppingCriterionThreshold = 0.001,
  regularizationCoefficient = 100, 
  regularizer = Regularizer.NONE,
  optimizationAlgorithm = OptimizationAlgorithm.GRADIENT_DESCENT,
  positiveDevSetSize = 0, 
  negativeDevSetSize = 0,
  persistWeightsAtPrefix = outputPrefix)

# now, do LOO cross validation
labeledExamples = logReg.ReadExamples(labeledFeaturesFilename)
(precision, recall, accuracy, f1) = logReg.NFoldCrossValidation(labeledExamples, 19, learningInfo, 0.5)
print 'precision:\t{0}\nrecall:\t{1}\naccuracy:\t{2}\nf1:   \t{3}\n'.format(precision, recall, accuracy, f1)

exit(0);

# for each unlabeled example in raw/docId-randomLabel, write a line in features/randomLabeled.txt.
unlabeledFeaturesFilename = '{0}.unlabeled'.format(outputPrefix)
if not os.path.exists('{0}/docId-randomLabel.txt'.format(rawDir)):
  WriteFeaturesFile('{0}/docId-randomLabel.txt'.format(rawDir), unlabeledFeaturesFilename)

# then, read unlabeledExamples
unlabeledExamples = logReg.ReadExamples(unlabeledFeaturesFilename)

# leave-one-out with self learning
labeledExamples = list(logReg.examples)
for i in range(0, len(labeledExamples)):
  print 'now using {0} as a test example'.format(i)
  testExample = labeledExamples[i]
  seedExamples = labeledExamples[0:i]
  seedExamples.extend(labeledExamples[i+1:])
  SelfLearn(logReg, seedExamples, unlabeledExamples, testExample)

