"""Card, shoe and hand-value primitives for the Blackjack simulator.

A card is represented as a (rank_label, value) tuple, e.g. ('10', 10) or
('A', 11). Grouping all ten-valued cards (10/J/Q/K) under the single rank
label '10' is a common simplification in basic-strategy engines: it keeps
the pair-splitting state space small (10 pair classes instead of 13) while
still reflecting real-table play, since 10/J/Q/K are strategically
identical in Blackjack.
"""
import random

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

# Hi-Lo card counting tags (Thorp-style point count), used for scenarios
# that involve the "Complete Point-Count System".
HI_LO_TAGS = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "A": -1,
}


def hand_value(values):
    """Best total (<=21 if possible) and whether the hand is 'soft'.

    values: list of ints, aces already encoded as 11.
    Returns (total, is_soft) where is_soft means at least one ace is
    still being counted as 11 in the returned total.
    """
    total = sum(values)
    soft_aces = values.count(11)
    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1
    return total, soft_aces > 0


class Shoe:
    """A multi-deck shoe with Hi-Lo running count tracking.

    Card visibility to the counting system is caller-controlled via
    `reveal()` — this matters because a dealer's hole card must not affect
    the running count until it is actually turned over, otherwise the
    learned policy would be conditioning on information it could not see
    at decision time.
    """

    def __init__(self, num_decks=6, penetration=0.75, rng=None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = rng or random.Random()
        self._build_and_shuffle()

    def _build_and_shuffle(self):
        single_deck = []
        for r in ["2", "3", "4", "5", "6", "7", "8", "9"]:
            single_deck += [(r, int(r))] * 4
        single_deck += [("10", 10)] * 16
        single_deck += [("A", 11)] * 4
        self.cards = single_deck * self.num_decks
        self.rng.shuffle(self.cards)
        self.cut_index = int(len(self.cards) * self.penetration)
        self.pos = 0
        self.running_count = 0

    def needs_shuffle(self):
        return self.pos >= self.cut_index

    def draw(self):
        if self.pos >= len(self.cards):
            # Extremely rare (very long streak of hits/splits); reshuffle
            # rather than crash. Slightly breaks count purity for that one
            # round only.
            self._build_and_shuffle()
        card = self.cards[self.pos]
        self.pos += 1
        return card

    def reveal(self, card):
        """Register a card as seen by the counting system."""
        self.running_count += HI_LO_TAGS[card[0]]

    def decks_remaining(self):
        return max((len(self.cards) - self.pos) / 52.0, 0.5)

    def true_count(self):
        return self.running_count / self.decks_remaining()

    def reshuffle_if_needed(self):
        if self.needs_shuffle():
            self._build_and_shuffle()
