import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        return 0

    def isEnd(self, state):
        return state == len(self.query)

    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        for newState in range(state+1, len(self.query)+1):
            result.append((self.query[state:newState], newState, self.unigramCost(self.query[state:newState])))
        return result

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    result = []
    for word in ucs.actions:
        result.append(word)
    return ' '.join(result)

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        return (0, wordsegUtil.SENTENCE_BEGIN)

    def isEnd(self, state):
        index, prevWord = state
        return index == len(self.queryWords)

    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        index, prevWord = state
        nextWords = self.possibleFills(self.queryWords[index])
        if len(nextWords) == 0:
            nextWords = [self.queryWords[index]]
        for word in nextWords:
            result.append((word, (index+1, word), self.bigramCost(prevWord, word)))
        return result

def insertVowels(queryWords, bigramCost, possibleFills):
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    result = []
    for word in ucs.actions:
        result.append(word)
    return ' '.join(result)

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        return (0, wordsegUtil.SENTENCE_BEGIN)

    def isEnd(self, state):
        index, prevWord = state
        return index == len(self.query)

    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        index, prevWord = state
        for i in range(index+1, len(self.query)+1):
            for word in self.possibleFills(self.query[index:i]):
                result.append((word, (i, word), self.bigramCost(prevWord, word)))
        return result

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    result = []
    for word in ucs.actions:
        result.append(word)
    return ' '.join(result)

############################################################

if __name__ == '__main__':
    shell.main()
