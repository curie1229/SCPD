import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        return '0'

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        return ['action']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        res = []
        if state == '0':
            res.append(('1', 0.9, 1))
            res.append(('2', 0.1, 99))
        return res

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        return 1

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        handValue, peekInd, deck = state
        res = []
        if deck == None:
            return []
        # Take
        if action == 'Take':
            if peekInd == None:
                for i in range(0, len(deck)):
                    if deck[i] <= 0:
                        continue
                    deckList = list(deck)
                    prob = deck[i]/sum(deckList)
                    deckList[i] -= 1
                    newHandValue = handValue + self.cardValues[i]
                    # burst
                    if newHandValue > self.threshold:
                        res.append(((newHandValue, None, None), prob, 0))
                    # not burst
                    elif sum(deckList) > 0:
                        res.append(((newHandValue, None, tuple(deckList)), prob, 0))
                    else:
                        res.append(((newHandValue, None, None), prob, newHandValue))

            # take peeked card
            else:
                deckList = list(deck)
                deckList[peekInd] -= 1
                newHandValue = handValue + self.cardValues[peekInd]
                # burst
                if newHandValue > self.threshold:
                    res.append(((newHandValue, None, None), 1, 0))
                # not burst and not run out of deck
                elif sum(deckList) > 0:
                    res.append(((newHandValue, None, tuple(deckList)), 1, 0))
                # not burst but run out of deck
                else:
                    res.append(((newHandValue, None, None), 1, newHandValue))
        # Peek
        if action == 'Peek':
            if peekInd != None:
                return []
            for i in range(0, len(deck)):
                res.append(((handValue, i, deck), deck[i]/sum(list(deck)), -self.peekCost))
        # Quit
        if action == 'Quit':
            res.append(((0, None, None), 1, handValue))
        return res

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    return BlackjackMDP(cardValues=[15, 1, 5], multiplicity=10,
                                              threshold=20, peekCost=1)

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        V = max([self.getQ(newState,newAction) for newAction in self.actions(newState)])
        Q = self.getQ(state,action)
        gradient = self.getStepSize() * (Q - (reward + self.discount * V))
        for f, v in self.featureExtractor(state,action):
            self.weights[f] -= gradient * v

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),featureExtractor,0.2)
    util.simulate(mdp, rl, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
    rl.explorationProb = 0.0
    vi = ValueIteration()
    vi.solve(mdp)
    cnt = 0
    for state in mdp.states:
        if rl.getAction(state) != vi.pi[state]:
            cnt += 1
    print(cnt, len(mdp.states), cnt/len(mdp.states))

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    if state == None:
        return []
    total, nextCard, counts = state
    features = []
    features.append(((action, total), 1))
    if counts != None:
        countsList=list(counts)
        for i in range(0, len(counts)):
            featureKey = (action, i, counts[i])
            featureValue = 1
            features.append((featureKey, featureValue))
            if counts[i] > 0:
                countsList[i] = 1
        featureKey = (action,tuple(countsList))
        featureValue = 1
        features.append((featureKey,featureValue))
    return features

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    vi = ValueIteration()
    vi.solve(originalMDP)
    rl1 = util.FixedRLAlgorithm(vi.pi.copy())
    util.simulate(modified_mdp, rl1, numTrials=10000, maxIterations=1000, verbose=True, sort=False)

    rl2 = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, 0.2)
    util.simulate(modified_mdp, rl2, numTrials=10000, maxIterations=1000, verbose=True, sort=False)