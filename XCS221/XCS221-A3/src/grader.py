#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import util

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

class AddNoiseMDP(util.MDP):
    def __init__(self, originalMDP):
        self.originalMDP = originalMDP

    def startState(self):
        return self.originalMDP.startState()

    # Return set of actions possible from |state|.
    def actions(self, state):
        return self.originalMDP.actions(state)

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        originalSuccAndProbReward = self.originalMDP.succAndProbReward(state, action)
        newSuccAndProbReward = []
        for state, prob, reward in originalSuccAndProbReward:
            newProb = 0.5 * prob + 0.5 / len(originalSuccAndProbReward)
            newSuccAndProbReward.append((state, newProb, reward))
        return newSuccAndProbReward

    # Return set of actions possible from |state|.
    def discount(self):
        return self.originalMDP.discount()

#########
# TESTS #
#########

class Test_2a(GradedTestCase):
  @graded(is_hidden=True)
  def test_hidden(self):
    """2a-hidden-0:  Hidden test for CounterexampleMDP. Ensure that V[startState] is greater after noise is added."""
    mdp = submission.CounterexampleMDP()
    mdp.computeStates()
    algorithm = submission.ValueIteration()
    algorithm.solve(mdp, .001)
    originalVal = algorithm.V[mdp.startState()]
    mdp = AddNoiseMDP(mdp)
    mdp.computeStates()
    algorithm.solve(mdp, .001)
    newVal = algorithm.V[mdp.startState()]
    # BEGIN_HIDE
    # END_HIDE

class Test_3a(GradedTestCase):
  @graded()
  def test_basic(self):
    """3a-basic-0:  Basic test for succAndProbReward() that covers several edge cases."""
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take')
    ]
    for gold, mdp, state, action in tests:
      # Feel free to uncomment this lines if you'd like to print out states/actions
      # print(('   state: {}, action: {}'.format(state, action)))
      self.assertEqual(gold, mdp.succAndProbReward(state, action))

  @graded(is_hidden=True)
  def test_hidden(self):
    """3a-hidden-0:  Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct."""
    mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                  threshold=40, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)
    # BEGIN_HIDE
    # END_HIDE

class Test_3b(GradedTestCase):

  @graded()
  def test_basic(self):
    """3b-basic-0:  Test for peekingMDP().  Ensure that in at least 10% of states, the optimal policy is to peek."""
    mdp = submission.peekingMDP()
    vi = submission.ValueIteration()
    vi.solve(mdp)
    self.assertEqual(mdp.threshold, 20)
    self.assertEqual(mdp.peekCost, 1)
    f = len([a for a in list(vi.pi.values()) if a == 'Peek']) / float(len(list(vi.pi.values())))
    self.assertLess(.1, f)
    # Feel free to uncomment these lines if you'd like to print out states/actions
    # for k, v in vi.pi.iteritems():
    #     print k, v

class Test_4a(GradedTestCase):

  @graded(timeout=10)
  def test_basic(self):
    """4a-basic-0:  Basic test for incorporateFeedback() using NumberLineMDP."""
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.identityFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback(0, 1, 0, 1)
    self.assertEqual(0, rl.getQ(0, -1))
    self.assertEqual(0, rl.getQ(0, 1))

    rl.incorporateFeedback(1, 1, 1, 2)
    self.assertEqual(0, rl.getQ(0, -1))
    self.assertEqual(0, rl.getQ(0, 1))
    self.assertEqual(0, rl.getQ(1, -1))
    self.assertEqual(1, rl.getQ(1, 1))

    rl.incorporateFeedback(2, -1, 1, 1)
    self.assertEqual(1.9, rl.getQ(2, -1))
    self.assertEqual(0, rl.getQ(2, 1))

  @graded(timeout=3, is_hidden=True)
  def test_hidden(self):
    """4a-hidden-0:  Hidden test for incorporateFeedback(). Run QLearningAlgorithm on smallMDP, then ensure that getQ returns reasonable value."""
    smallMDP = self.run_with_solution_if_possible(submission,
                                                  lambda sub_or_sol: sub_or_sol.BlackjackMDP(cardValues=[1,5], multiplicity=2, threshold=10, peekCost=1))
    smallMDP.computeStates()
    rl = submission.QLearningAlgorithm(smallMDP.actions, smallMDP.discount(),
                                   submission.identityFeatureExtractor,
                                   0.2)
    util.simulate(smallMDP, rl, 30000)
    # BEGIN_HIDE
    # END_HIDE


# NOTE: this is not a true "test" for grading purposes -- it's worth zero points.  This function exists to help you
# as you're working on question 4b; this question requires a written response on the assignment, but you will need
# to run some code to get the stats that will go into your answer.  Check out the partial implementation of the
# 'simulate_QL_over_MDP' function in submission.py to see one place where you might consider printing these stats.
class Test_4b(GradedTestCase):

  @graded(timeout=60)
  def test_helper(self):
    """4b-helper-0:  Helper function to run Q-learning simulations for question 4b."""
    submission.simulate_QL_over_MDP(submission.smallMDP, submission.identityFeatureExtractor)
    submission.simulate_QL_over_MDP(submission.largeMDP, submission.identityFeatureExtractor)
    # NOTE:  This is bad unit testing practice- the course staff is including
    # always-skipped tests to make the test suite a one-stop shop for students.
    # Production unit tests, in general, should not include test cases that
    # always skip. Usually skipped tests are based on version number, available
    # resources, etc.  For example, this library skips hidden tests on student
    # machines because the solution resources are not available on those
    # machines.
    self.skipTest("This test case is a helper function for students.")

class Test_4c(GradedTestCase):
  @graded(timeout=10)
  def test_basic(self):
    """4c-basic-0:  Basic test for blackjackFeatureExtractor.  Runs QLearningAlgorithm using blackjackFeatureExtractor, then checks to see that Q-values are correct."""
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.blackjackFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))
    self.assertEqual(28, rl.getQ((7, None, (0, 1)), 'Quit'))
    self.assertEqual(7, rl.getQ((7, None, (1, 0)), 'Quit'))
    self.assertEqual(14, rl.getQ((2, None, (0, 2)), 'Quit'))
    self.assertEqual(0, rl.getQ((2, None, (0, 2)), 'Take'))

class Test_4d(GradedTestCase):
  @graded(timeout=60)
  def test_helper(self):
    """4d-helper-0:  Helper function to compare rewards when simulating RL over two different MDPs in question 4d."""
    submission.compare_changed_MDP(submission.originalMDP, submission.newThresholdMDP, submission.blackjackFeatureExtractor)
    # NOTE:  This is bad unit testing practice- the course staff is including
    # always-skipped tests to make the test suite a one-stop shop for students.
    # Production unit tests, in general, should not include test cases that
    # always skip. Usually skipped tests are based on version number, available
    # resources, etc.  For example, this library skips hidden tests on student
    # machines because the solution resources are not available on those
    # machines.
    self.skipTest("This test case is a helper function for students.")

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