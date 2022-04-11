#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
from util import *

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

class Test_1a(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1a-0-basic:  Basic test case."""
    ans = {"a":2, "b":1}
    self.assertEqual(ans,
                     submission.extractWordFeatures("a b a"))

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """1a-1-hidden:  Test multiple instances of the same word in a sentence."""
    random.seed(42)
    for i in range(10):
      sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
      self.compare_with_solution_or_wait(submission, 'extractWordFeatures', lambda f: f(sentence))
class Test_1b(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1b-0-basic:  Basic sanity check for learning correct weights on two training and testing examples each."""
    trainExamples = (("hello world", 1), ("goodnight moon", -1))
    testExamples = (("hello", 1), ("moon", -1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    self.assertLess(0, weights["hello"])
    self.assertGreater(0, weights["moon"])

  @graded(timeout=1)
  def test_1(self):
    """1b-1-basic:  Test correct overriding of positive weight due to one negative instance with repeated words."""
    trainExamples = (("hi bye", 1), ("hi hi", -1))
    testExamples = (("hi", -1), ("bye", 1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    self.assertGreater(0, weights["hi"])
    self.assertLess(0, weights["bye"])

  @graded(timeout=8)
  def test_2(self):
    """1b-2-basic:  Test classifier on real polarity dev dataset."""
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, dev error = %s" % (trainError, devError)))
    self.assertGreater(0.04, trainError)
    self.assertGreater(0.30, devError)
class Test_1c(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1c-0-basic:  test correct generation of random dataset labels"""
    weights = {"hello":1, "world":1}
    data = submission.generateDataset(5, weights)
    for datapt in data:
        self.assertEqual((dotProduct(datapt[0], weights) >= 0), (datapt[1] == 1))

  @graded(timeout=1)
  def test_1(self):
    """1c-1-basic:  test that the randomly generated example actually coincides with the given weights"""
    weights = {}
    for i in range(100):
      weights[str(i + 0.1)] = 1
    data = submission.generateDataset(100, weights)
    for datapt in data:
      self.assertEqual(False, dotProduct(datapt[0], weights) == 0)
class Test_1e(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1e-0-basic:  test basic character n-gram features"""
    fe = submission.extractCharacterFeatures(3)
    sentence = "hello world"
    ans = {"hel":1, "ell":1, "llo":1, "low":1, "owo":1, "wor":1, "orl":1, "rld":1}
    self.assertEqual(ans, fe(sentence))

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """1e-1-hidden:  test feature extraction on repeated character n-grams"""
    random.seed(42)
    for i in range(10):
      sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
      for n in range(1, 4):
        self.compare_with_solution_or_wait(submission, 'extractCharacterFeatures', lambda f: f(n)(sentence))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)