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
from log_reg import LogisticRegression

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

# initialize the feature extractor
extractor = FeatureExtractor('|')

# for each labeled example in raw/docId-label, write a line in features/labeled.txt. Instead of the site-ID in the original file, write the non-zero feature IDs and their values.
labeledFeaturesFilename = '{0}.labeled'.format(outputPrefix)
WriteFeaturesFile('{0}/docId-label.txt'.format(rawDir), labeledFeaturesFilename)

# now, do LOO cross validation
logReg = LogisticRegression(labeledFeaturesFilename, learningRate = 0.001, maxIterLossDiff = 0.0001, regularizationConst = 0)
(precision, recall, accuracy, f1) = logReg.NFoldCrossValidation(20, outputPrefix, 'SGD')
print 'precision:\t{0}\nrecall:\t{1}\naccuracy:\t{2}\nf1:   \t{3}\n'.format(precision, recall, accuracy, f1)

exit(0);
logReg.MinimizeLossWithSGD(logReg.examples, persistWeightsAt='{0}.optimalFeatureWeightsWithLabeledExamples'.format(outputPrefix))

# for each unlabeled example in raw/docId-randomLabel, write a line in features/randomLabeled.txt.
unlabeledFeaturesFilename = '{0}.unlabeled'.format(outputPrefix)
WriteFeaturesFile('{0}/docId-randomLabel.txt'.format(rawDir), unlabeledFeaturesFilename)

# find unlabeled examples of which we are confident
unlabeledExamples = logReg.ReadExamples(unlabeledFeaturesFilename)
selfLearntExamples = []
for unlabeledExampleId in range(0, len(unlabeledExamples)):
  prob1 = logReg.ComputeProb1(unlabeledExamples[unlabeledExampleId])
  if prob1 > 0.8:
    print 'adding {0}. prob1 = {1}'.format(unlabeledExampleId, prob1)
    newExample = unlabeledExamples[unlabeledExampleId]
    newExample.label = 1
    selfLearntExamples.append(newExample)
  elif prob1 < 0.01:
    print 'adding {0}. prob1 = {1}'.format(unlabeledExampleId, prob1)
    newExample = unlabeledExamples[unlabeledExampleId]
    newExample.label = 0
    selfLearntExamples.append(newExample)

# train the log linear model with self learnt examples
print '=============='
print '# original examples = {0}'.format(len(logReg.examples))
labeledExamples = list(logReg.examples)
logReg.examples = selfLearntExamples
print '# total examples = {0}'.format(len(logReg.examples))
logReg.MinimizeLossWithSGD(logReg.examples, persistWeightsAt='')

truePositives = 0
trueNegatives = 0
falsePositives = 0
falseNegatives = 0
for labeledExample in labeledExamples:
  if logReg.ComputeProb1(labeledExample) > 0.5:
    if labeledExample.label == 1:
      truePositives += 1.0
    else:
      falsePositives += 1.0
  else:
    if labeledExample.label == 1:
      falseNegatives += 1.0
    else:
      trueNegatives += 1.0

precision = truePositives / (truePositives + falsePositives)
recall = truePositives / (truePositives + falseNegatives)
accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
f1 = (precision + recall) / (precision * recall)
print 'precision:\t{0}\nrecall:\t{1}\naccuracy:\t{2}\nf1:   \t{3}\n'.format(precision, recall, accuracy, f1)
