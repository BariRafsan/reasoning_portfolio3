from dataclasses import dataclass


@dataclass
class Rules:
    num_decks: int = 6
    dealer_hits_soft17: bool = True
    blackjack_payout: float = 1.5  # 3:2
    double_after_split: bool = True
    resplit_aces: bool = False
    hit_split_aces: bool = False
    max_hands: int = 4
    surrender_allowed: bool = False
    penetration: float = 0.75


BASE_RULES = Rules()

RULESETS = {
    "base": Rules(),
    # Variant A: dealer stands on soft 17 (favors the player).
    "dealer_stands_soft17": Rules(dealer_hits_soft17=False),
    # Variant B: 6:5 blackjack payout instead of 3:2 (favors the house,
    # a well-known modern casino rule change condemned by advantage players).
    "blackjack_6to5": Rules(blackjack_payout=1.2),
    # Variant C: late surrender allowed.
    "surrender_allowed": Rules(surrender_allowed=True),
    # Variant D: single deck (much stronger counting signal).
    "single_deck": Rules(num_decks=1, penetration=0.5),
}
