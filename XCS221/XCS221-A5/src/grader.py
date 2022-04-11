#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
from graderUtil import graded, CourseTestRunner, GradedTestCase

import util

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def verify_schedule(bulletin, profile, schedule, checkUnits = True):
    """
    Returns true if the schedule satisifies all requirements given by the profile.
    """
    goodSchedule = True
    all_courses_taking = dict((s[1], s[0]) for s in schedule)

    # No course can be taken twice.
    goodSchedule *= len(all_courses_taking) == len(schedule)
    if not goodSchedule:
        print('course repeated')
        return False

    # Each course must be offered in that quarter.
    goodSchedule *= all(bulletin.courses[s[1]].is_offered_in(s[0]) for s in schedule)
    if not goodSchedule:
        print('course not offered')
        return False

    # If specified, only take the course at the requested time.
    for req in profile.requests:
        if len(req.quarters) == 0: continue
        goodSchedule *= all([s[0] in req.quarters for s in schedule if s[1] in req.cids])
    if not goodSchedule:
        print('course taken at wrong time')
        return False

    # If a request has multiple courses, at most one is chosen.
    for req in profile.requests:
        if len(req.cids) == 1: continue
        goodSchedule *= len([s for s in schedule if s[1] in req.cids]) <= 1
    if not goodSchedule:
        print('more than one exclusive group of courses is taken')
        return False

    # Must take a course after the prereqs
    for req in profile.requests:
        if len(req.prereqs) == 0: continue
        cids = [s for s in schedule if s[1] in req.cids] # either empty or 1 element
        if len(cids) == 0: continue
        quarter, cid, units = cids[0]
        for prereq in req.prereqs:
            if prereq in profile.taking:
                goodSchedule *= prereq in all_courses_taking
                if not goodSchedule:
                    print('not all prereqs are taken')
                    return False
                goodSchedule *= profile.quarters.index(quarter) > \
                    profile.quarters.index(all_courses_taking[prereq])
    if not goodSchedule:
        print('course is taken before prereq')
        return False

    if not checkUnits: return goodSchedule
    # Check for unit loads
    unitCounters = collections.Counter()
    for quarter, c, units in schedule:
        unitCounters[quarter] += units
    goodSchedule *= all(profile.minUnits <= u and u <= profile.maxUnits \
        for k, u in list(unitCounters.items()))
    if not goodSchedule:
        print('unit count out of bound for quarter')
        return False

    return goodSchedule

# Load all courses.
bulletin = util.CourseBulletin('courses.json')

#########
# TESTS #
#########

class Test_0c(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """0c-0-basic:  Basic test for create_chain_csp."""
    solver = submission.BacktrackingSearch()
    solver.solve(submission.create_chain_csp(4))
    self.assertEqual(1, solver.optimalWeight)
    self.assertEqual(2, solver.numOptimalAssignments)
    self.assertEqual(9, solver.numOperations)
class Test_1a(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1a-0-basic:  Basic test for create_nqueens_csp for n=8."""
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    self.assertEqual(1.0, nQueensSolver.optimalWeight)
    self.assertEqual(92, nQueensSolver.numOptimalAssignments)
    self.assertEqual(2057, nQueensSolver.numOperations)

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """1a-1-hidden:  Test create_nqueens_csp with n=3."""
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(3))
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_2(self):
    """1a-2-hidden:  Test create_nqueens_csp with different n."""
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(4))
    # BEGIN_HIDE
    # END_HIDE

    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(7))
    # BEGIN_HIDE
    # END_HIDE
class Test_1b(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1b-0-basic:  Basic test for MCV with n-queens CSP."""
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    self.assertEqual(1.0, mcvSolver.optimalWeight)
    self.assertEqual(92, mcvSolver.numOptimalAssignments)
    self.assertEqual(1361, mcvSolver.numOperations)

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """1b-1-hidden:  Test for MCV with n-queens CSP."""
    mcvSolver = submission.BacktrackingSearch()
    nqueens_csp = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.create_nqueens_csp(8))
    mcvSolver.solve(nqueens_csp, mcv = True)
    # BEGIN_HIDE    
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_2(self):
    """1b-2-hidden:  Test MCV with different CSPs."""
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(util.create_map_coloring_csp(), mcv = True)
    # BEGIN_HIDE
    # END_HIDE
class Test_1c(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """1c-0-basic:  Basic test for AC-3 with n-queens CSP"""
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(submission.create_nqueens_csp(8), ac3 = True)
    self.assertEqual(1.0, acSolver.optimalWeight)
    self.assertEqual(92, acSolver.numOptimalAssignments)
    self.assertEqual(769, acSolver.numOperations)

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """1c-1-hidden:  Test AC-3 for map coloring CSP"""
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(util.create_map_coloring_csp(), ac3 = True)
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_2(self):
    """1c-2-hidden:  Test MCV+AC-3 for n-queens CSP with n=8"""
    acSolver = submission.BacktrackingSearch()
    nqueens_csp = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.create_nqueens_csp(8))
    acSolver.solve(nqueens_csp, mcv=True, ac3=True)
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_3(self):
    """1c-3-hidden:  Test MCV+AC-3 for n-queens CSP with n=7"""
    acSolver = submission.BacktrackingSearch()
    nqueens_csp = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.create_nqueens_csp(7))
    acSolver.solve(nqueens_csp, mcv=True, ac3=True)
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_4(self):
    """1c-4-hidden:  Test MCV+AC-3 for map coloring CSP"""
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(util.create_map_coloring_csp(), mcv = True, ac3 = True)
    # BEGIN_HIDE
    # END_HIDE
class Test_2b(GradedTestCase):
  @graded(timeout=1)
  def test_0(self):
    """2b-0-basic:  Basic test for get_sum_variable"""
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    csp.add_unary_factor(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    self.assertEqual(4, sumSolver.numOptimalAssignments)

    csp.add_unary_factor(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    self.assertEqual(2, sumSolver.numOptimalAssignments)

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """2b-1-hidden:  Test get_sum_variable with empty list of variables."""
    csp = util.CSP()
    sumVar = submission.get_sum_variable(csp, 'zero', [], 15)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    # BEGIN_HIDE
    # END_HIDE

    csp.add_unary_factor(sumVar, lambda n: n > 0)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=1, is_hidden=True)
  def test_2(self):
    """2b-2-hidden:  Test get_sum_variable with different variables."""
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2])
    csp.add_variable('B', [0, 1, 2])
    csp.add_variable('C', [0, 1, 2])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-7', ['A', 'B', 'C'], 7)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    # BEGIN_HIDE
    # END_HIDE

    csp.add_unary_factor(sumVar, lambda n: n == 6)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    # BEGIN_HIDE
    # END_HIDE
class Test_3a(GradedTestCase):
  @graded(timeout=4)
  def test_0(self):
    """3a-0-basic:  Basic test for add_quarter_constraints"""
    profile = util.Profile(bulletin, 'profile3a.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    self.assertEqual(3, alg.numOptimalAssignments)
    sol = util.extract_course_scheduling_solution(profile, alg.optimalAssignment)
    for assignment in alg.allAssignments:
      sol = util.extract_course_scheduling_solution(profile, assignment)
      self.assertTrue(verify_schedule(bulletin, profile, sol, False))

  @graded(timeout=3, is_hidden=True)
  def test_1(self):
    """3a-1-hidden:  Test add_quarter_constraints with different profiles."""
    profile = util.Profile(bulletin, 'profile3a1.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)
    # Verify correctness.
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=3, is_hidden=True)
  def test_2(self):
    """3a-2-hidden:  Test add_quarter_constraints with no quarter specified"""
    profile = util.Profile(bulletin, 'profile3a2.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    # BEGIN_HIDE
    # END_HIDE
class Test_3b(GradedTestCase):
  @graded(timeout=7)
  def test_0(self):
    """3b-0-basic:  Basic test for add_unit_constraints"""
    profile = util.Profile(bulletin, 'profile3b.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_unit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    self.assertEqual(15, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
      sol = util.extract_course_scheduling_solution(profile, assignment)
      self.assertTrue(verify_schedule(bulletin, profile, sol))

  @graded(timeout=3, is_hidden=True)
  def test_1(self):
    """3b-1-hidden:  Test add_unit_constraints with different profiles"""
    profile = util.Profile(bulletin, 'profile3b1.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_unit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=4, is_hidden=True)
  def test_2(self):
    """3b-2-hidden:  Test unsatisfiable scheduling"""
    profile = util.Profile(bulletin, 'profile3b2.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_all_additional_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)
    # Verify correctness.
    # BEGIN_HIDE
    # END_HIDE

  @graded(timeout=25, is_hidden=True)
  def test_3(self):
    """3b-3-hidden:  Test MVC+AC-3+all additional constraints"""
    profile = util.Profile(bulletin, 'profile3b3.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_all_additional_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp, mcv = True, ac3 = True)

    # Verify correctness.
    # BEGIN_HIDE
    # END_HIDE


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