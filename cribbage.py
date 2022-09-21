from enum import Enum

class Suit(Enum):
    SPADES = 1
    CLUBS = 2
    DIAMONDS = 3
    HEARTS = 4

class Rank(Enum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

class Score:
    def __init__(self):
        self.combinations = []

    def total(self):
        return sum([combo.score for combo in self.combinations])

class ScoringCombination:
    def __init__(self, cards, score):
        self.cards = cards
        self.score = score

def rank(card):
    rank, _ = card
    return rank

def suit(card):
    _, suit = card
    return suit

def calculateScore(starter, hand):
    if len(hand) != 4 or starter is None:
        raise RuntimeError("A hand must include 4 cards and a starter in order to be scored.")
    score = Score()
    score.combinations += scoreHeels(starter, hand)
    score.combinations += scoreFlushes(starter, hand)
    return score

def scoreFlushes(starter, hand):
    testSuit = suit(hand[0])
    if all(cardSuit.value == testSuit.value for cardSuit in list(map(suit, hand))):
        if suit(starter).value == testSuit.value:
            return [ScoringCombination(hand + [starter], 5)]
        else:
            return [ScoringCombination(hand, 4)]
    return []

def scoreHeels(starter, hand):
    starterSuit = suit(starter)
    matchingJacks = [(cardRank, cardSuit) for (cardRank, cardSuit) in hand if cardRank == Rank.JACK and cardSuit.value == starterSuit.value]
    if len(matchingJacks) == 1:
        return [ScoringCombination(matchingJacks[0], 1)]
    return []