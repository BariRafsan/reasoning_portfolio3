"""Round-level Blackjack simulation.

A "round" = one shuffle-aware deal, the player's decisions (possibly across
several hands if splits occur), the dealer's fixed-strategy play, and
settlement. The module exposes `play_round`, which is algorithm-agnostic:
it is driven by any object exposing `choose_action(state, avail_actions)`.

State encoding
---------------
A player hand's state is a tuple:
    (('hard', total) | ('soft', total) | ('pair', rank_label), dealer_upcard, tc_bucket)
`tc_bucket` is None for non-counting scenarios (dropped from the tuple),
or an integer bucket of the Hi-Lo true count for counting scenarios.

Design simplifications (documented for the paper's methodology section):
  * All ten-valued cards (10/J/Q/K) share rank label '10', so "pair of 10s"
    covers any two ten-valued cards, matching common basic-strategy engines.
  * Split hands are treated as independent sub-episodes: each gets its own
    trajectory and its own terminal reward (profit / base bet), which keeps
    Monte-Carlo/Q-learning updates well-defined even though only a terminal
    reward exists per decision chain.
  * Split aces receive exactly one card and auto-stand unless
    `rules.hit_split_aces` is set (standard casino rule).
"""
from enum import IntEnum

from .cards import hand_value


class Action(IntEnum):
    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4


ACTION_NAMES = {a.value: a.name for a in Action}


def _state_kind(cards):
    values = [c[1] for c in cards]
    total, soft = hand_value(values)
    is_pair = len(cards) == 2 and cards[0][0] == cards[1][0]
    if is_pair:
        return ("pair", cards[0][0]), total, soft
    if soft:
        return ("soft", total), total, soft
    return ("hard", total), total, soft


def encode_state(cards, dealer_upcard_value, tc_bucket):
    kind, total, soft = _state_kind(cards)
    if tc_bucket is None:
        return (kind, dealer_upcard_value)
    return (kind, dealer_upcard_value, tc_bucket)


def bucket_true_count(tc, lo=-6, hi=6):
    return int(max(lo, min(hi, round(tc))))


def play_player_hands(player_cards, dealer_upcard_value, shoe, agent, rules, get_tc_bucket):
    """Plays out all of the player's hands (incl. splits).

    Returns a list of dicts, one per finished hand:
        {'trajectory': [(state, action), ...], 'cards': [...],
         'bet_mult': int, 'busted': bool, 'surrendered': bool}
    """
    results = []
    pending = [{
        "cards": list(player_cards),
        "bet_mult": 1,
        "first_action": True,
        "from_split": False,
        "trajectory": [],
    }]
    num_hands_active = 1

    while pending:
        h = pending.pop()
        cards = h["cards"]
        trajectory = h["trajectory"]

        while True:
            total, soft = hand_value([c[1] for c in cards])
            if total > 21:
                h["busted"] = True
                h["surrendered"] = False
                results.append(h)
                break

            is_pair = len(cards) == 2 and cards[0][0] == cards[1][0]
            can_double = (
                h["first_action"] and len(cards) == 2 and
                (not h["from_split"] or rules.double_after_split)
            )
            can_split = (
                h["first_action"] and is_pair and len(cards) == 2 and
                num_hands_active < rules.max_hands
            )
            can_surrender = (
                h["first_action"] and not h["from_split"] and
                len(cards) == 2 and rules.surrender_allowed
            )

            state = encode_state(cards, dealer_upcard_value, get_tc_bucket())
            avail = [Action.STAND, Action.HIT]
            if can_double:
                avail.append(Action.DOUBLE)
            if can_split:
                avail.append(Action.SPLIT)
            if can_surrender:
                avail.append(Action.SURRENDER)

            action = agent.choose_action(state, avail)
            trajectory.append((state, action))

            if action == Action.STAND:
                h["busted"] = False
                h["surrendered"] = False
                results.append(h)
                break

            if action == Action.SURRENDER:
                h["busted"] = False
                h["surrendered"] = True
                results.append(h)
                break

            if action == Action.HIT:
                card = shoe.draw()
                shoe.reveal(card)
                cards.append(card)
                h["first_action"] = False
                continue

            if action == Action.DOUBLE:
                card = shoe.draw()
                shoe.reveal(card)
                cards.append(card)
                h["bet_mult"] = 2
                new_total, _ = hand_value([c[1] for c in cards])
                h["busted"] = new_total > 21
                h["surrendered"] = False
                results.append(h)
                break

            if action == Action.SPLIT:
                c1, c2 = cards[0], cards[1]
                new1 = shoe.draw(); shoe.reveal(new1)
                new2 = shoe.draw(); shoe.reveal(new2)
                hand_a = {"cards": [c1, new1], "bet_mult": 1, "first_action": True,
                          "from_split": True, "trajectory": list(trajectory)}
                hand_b = {"cards": [c2, new2], "bet_mult": 1, "first_action": True,
                          "from_split": True, "trajectory": list(trajectory)}
                num_hands_active += 1

                splitting_aces = c1[0] == "A"
                if splitting_aces and not rules.hit_split_aces:
                    for hd in (hand_a, hand_b):
                        hd["busted"] = False
                        hd["surrendered"] = False
                        results.append(hd)
                elif splitting_aces and not rules.resplit_aces:
                    # Allow the one hit each but no further resplitting of aces.
                    for hd in (hand_a, hand_b):
                        hd["from_split"] = True
                        pending.append(hd)
                else:
                    pending.append(hand_a)
                    pending.append(hand_b)
                break
    return results


def play_dealer(dealer_cards, shoe, rules):
    shoe.reveal(dealer_cards[1])  # hole card now revealed
    while True:
        total, soft = hand_value([c[1] for c in dealer_cards])
        if total > 21:
            break
        if total < 17:
            card = shoe.draw(); shoe.reveal(card)
            dealer_cards.append(card)
            continue
        if total == 17 and soft and rules.dealer_hits_soft17:
            card = shoe.draw(); shoe.reveal(card)
            dealer_cards.append(card)
            continue
        break
    return dealer_cards


def play_round(shoe, rules, agent, get_tc_bucket_fn=None):
    """Plays one full round. Returns (hand_results, total_profit_units).

    total_profit_units is the round's profit measured in units of the base
    bet (i.e. multiply by the actual money bet to get currency profit).
    hand_results is the list from play_player_hands (with 'profit' added
    to each dict), needed by the learning agent to credit trajectories.
    """
    shoe.reshuffle_if_needed()

    if get_tc_bucket_fn is None:
        get_tc_bucket = lambda: None
    else:
        get_tc_bucket = lambda: bucket_true_count(shoe.true_count())

    player_cards = [shoe.draw(), shoe.draw()]
    dealer_cards = [shoe.draw(), shoe.draw()]
    shoe.reveal(player_cards[0]); shoe.reveal(player_cards[1])
    shoe.reveal(dealer_cards[0])  # upcard only; hole card revealed later

    dealer_upcard_value = dealer_cards[0][1]
    player_bj = hand_value([c[1] for c in player_cards])[0] == 21
    dealer_bj = hand_value([c[1] for c in dealer_cards])[0] == 21

    if player_bj or dealer_bj:
        shoe.reveal(dealer_cards[1])
        if player_bj and dealer_bj:
            profit = 0.0
        elif player_bj:
            profit = rules.blackjack_payout
        else:
            profit = -1.0
        return [], profit

    hand_results = play_player_hands(player_cards, dealer_upcard_value, shoe, agent, rules, get_tc_bucket)

    any_live = any((not h["busted"]) and (not h["surrendered"]) for h in hand_results)
    if any_live:
        dealer_cards = play_dealer(dealer_cards, shoe, rules)
    dealer_total, _ = hand_value([c[1] for c in dealer_cards])
    dealer_bust = dealer_total > 21

    total_profit = 0.0
    for h in hand_results:
        if h["surrendered"]:
            profit = -0.5
        elif h["busted"]:
            profit = -1.0 * h["bet_mult"]
        elif dealer_bust:
            profit = 1.0 * h["bet_mult"]
        else:
            p_total, _ = hand_value([c[1] for c in h["cards"]])
            if p_total > dealer_total:
                profit = 1.0 * h["bet_mult"]
            elif p_total < dealer_total:
                profit = -1.0 * h["bet_mult"]
            else:
                profit = 0.0
        h["profit"] = profit
        total_profit += profit

    return hand_results, total_profit
