#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    phi = {}
    words = x.split()
    for word in words:
    	if word not in phi:
    	    phi[word] = 0
    	phi[word] += 1
    return phi

############################################################
# Problem 1b: stochastic gradient descent
def sdF(w, i, phi, y, eta):
    x, y = phi[i], y[i]
    dotProduct = 0
    for key, value in x.items():
        dotProduct += w[key] * value

    if (dotProduct * y) < 1:
        for key, value in x.items():
            gradient = -1 * value * y
            w[key] = w[key] - eta * gradient

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight

    phi, y = [], []
    for trainExample in trainExamples:
        phi.append(featureExtractor(trainExample[0]))
        y.append(trainExample[1])

    for sentence in phi:
        for key in sentence.items():
            weights[key] = 0
    for t in range(numIters):
       indexList = list(range(len(y)))
       random.shuffle(indexList)
       for i in indexList:
           sdF(weights, i, phi, y, eta)
    return weights

############################################################
# Problem 1c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        phi = None
        y = None
        numWords = random.randint(1, len(weights.keys()))
        phi = {list(weights.keys())[random.randint(0, numWords - 1)] : random.randint(1, 100000) for _ in range(numWords)}
        dotProduct = 0
        for key, value in phi.items():
            dotProduct += weights[key] * phi[key]
        if dotProduct > 0:
            y = 1
        else:
            y = -1
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 1e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        x = x.replace(" ", "")
        phi = {}
        for i in range(0, len(x) - n + 1):
            if x[i: i + n] not in phi:
                phi[x[i: i + n] ] = 0
            phi[x[i: i + n]] += 1
        return phi
    return extract