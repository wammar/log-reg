# coding: utf-8

import os
import sys
import io
import codecs
import re
import math
from collections import defaultdict

class FeatureExtractor:
  def __init__(self, featureIdSeparator):
    self.features = defaultdict(float)
    self.featureIdSeparator = featureIdSeparator

  # given tokenized text, split on all whitespaces and update the extracted ngram count features accordingly. case sensitive.
  # doesn't add any additional tags (e.g. <start-of-doc> ...etc).
  def ExtractNgramCount(self, text, ngramOrder, featureIdPrefix):
    # validation
    assert(ngramOrder == int(ngramOrder) and ngramOrder > 0)

    # scan text tokens
    tokens = text.split()
    for i in range(ngramOrder - 1, len(tokens)):
      featureId = featureIdPrefix
      relativeTokenIndexes = range(0, ngramOrder)
      relativeTokenIndexes.reverse()
      for j in relativeTokenIndexes:
        featureId += self.featureIdSeparator + tokens[i-j]
      self.features[featureId] += 1.0

  # given tokenized text, split on all whitespaces and update the extracted ngram existence features accordingly. case sensitive.
  # doesn't add any additional tags (e.g. <start-of-doc> ...etc).
  def ExtractNgramPresence(self, text, ngramOrder, featureIdPrefix):
    # validation
    assert(ngramOrder == int(ngramOrder) and ngramOrder > 0)

    # scan text tokens
    tokens = text.split()
    for i in range(ngramOrder - 1, len(tokens)):
      featureId = featureIdPrefix
      relativeTokenIndexes = range(0, ngramOrder)
      relativeTokenIndexes.reverse()
      for j in relativeTokenIndexes:
        featureId += self.featureIdSeparator + tokens[i-j]
      self.features[featureId] = 1.0

  # this feature characterizes a bias in the training set towards one of the classes
  def BiasFeature(self, featureIdPrefix):
    self.features[featureIdPrefix] = 1.0

