from itertools import groupby
from operator import itemgetter
from game import Card, Deck # Import from our new game.py


class Player:
    def __init__(self, pid, game):
        self.pid = pid
        self.game = game
        self.selected_card = None  # stores card AND position of card
        self.swap = None
        self.hand = self.deal_hand(game)
        self.deadwood = 0
        self.sets = []
        self.runs = []
        self.score = 0
        self.ready = False # Kept for compatibility with Conditions

    def highest_playable_card(self):
        """
        Returns the highest ranked card not in a set or a run
        :return: card object
        """
        # This function is no longer critical as update_melds_and_deadwood
        # is the source of truth, but we'll keep it.
        self.update_melds_and_deadwood()
        
        melded_cards = set(self.sets + self.runs)
        leftovers = [c for c in self.hand if c not in melded_cards]
        
        try:
            max_card = max(leftovers, key=lambda x: self._get_deadwood_val(x))
        except ValueError:
            max_card = Card(rank='0', suit='0')

        return max_card

    # --- NEW HELPER FUNCTIONS ---
    
    def _get_deadwood_val(self, card):
        rank = int(card.rank)
        if rank == 1: return 1
        if rank >= 10: return 10
        return rank

    def _get_deadwood_score(self, hand):
        """Calculates the deadwood score of a given hand."""
        return sum(self._get_deadwood_val(card) for card in hand)

    def _get_all_possible_melds(self, hand):
        """Finds all 3/4-card sets and 3+ card runs in a hand."""
        melds = []
        hand_tuple = tuple(sorted(hand))

        # 1. Find Sets
        h_sorted_rank = sorted(hand_tuple, key=lambda x: int(x.rank))
        for k, g in groupby(h_sorted_rank, lambda x: x.rank):
            group = list(g)
            if len(group) == 3:
                melds.append(tuple(group))
            elif len(group) == 4:
                # Add the 4-card set
                melds.append(tuple(group))
                # Add all 3-card subsets
                melds.append((group[0], group[1], group[2]))
                melds.append((group[0], group[1], group[3]))
                melds.append((group[0], group[2], group[3]))
                melds.append((group[1], group[2], group[3]))

        # 2. Find Runs
        h_sorted_suit = sorted(hand_tuple, key=lambda x: (x.suit, int(x.rank)))
        for suit, suit_cards in groupby(h_sorted_suit, key=lambda x: x.suit):
            cards = list(suit_cards)
            
            # Get unique-ranked cards for this suit
            unique_cards_in_suit = []
            seen_ranks = set()
            for card in cards:
                rank = int(card.rank)
                if rank not in seen_ranks:
                    unique_cards_in_suit.append(card)
                    seen_ranks.add(rank)
            
            if len(unique_cards_in_suit) < 3:
                continue
                
            # Iterate through all possible sub-lists of 3+ cards
            for i in range(len(unique_cards_in_suit)):
                for j in range(i + 2, len(unique_cards_in_suit)):
                    run_candidate = unique_cards_in_suit[i:j+1]
                    
                    if len(run_candidate) < 3:
                        continue
                        
                    is_a_run = True
                    for k in range(len(run_candidate) - 1):
                        rank_a = int(run_candidate[k].rank)
                        rank_b = int(run_candidate[k+1].rank)
                        
                        # Handle A-2-3 (Ace is rank 1)
                        if k == 0 and rank_a == 1 and rank_b == 2:
                            continue
                        
                        # Handle all other ranks
                        if rank_b != rank_a + 1:
                            is_a_run = False
                            break
                    
                    if is_a_run:
                        melds.append(tuple(run_candidate))
                        
        return list(set(melds)) # Return unique melds

    def _find_best_meld_combination(self, hand_tuple):
        """
        Recursive helper to find the best melds.
        Returns (best_deadwood_score, list_of_best_melds)
        """
        if not hand_tuple:
            return 0, []
        
        if hand_tuple in self.memo:
            return self.memo[hand_tuple]
        
        hand = list(hand_tuple)
        
        # Base case: No melds, just return deadwood of current hand
        best_dw = self._get_deadwood_score(hand)
        best_melds = []
        
        all_possible_melds = self._get_all_possible_melds(hand)
        
        for meld in all_possible_melds:
            # Create the remaining hand after using this meld
            remaining_hand_list = list(hand)
            meld_cards = list(meld)
            
            valid_meld = True
            temp_hand = list(remaining_hand_list)
            for card_in_meld in meld_cards:
                if card_in_meld in temp_hand:
                    temp_hand.remove(card_in_meld)
                else:
                    valid_meld = False
                    break
            
            if not valid_meld:
                continue

            remaining_hand_tuple = tuple(sorted(temp_hand, key=lambda x: (x.suit, int(x.rank))))
            
            # Recurse
            dw_score, other_melds = self._find_best_meld_combination(remaining_hand_tuple)
            
            if dw_score < best_dw:
                best_dw = dw_score
                best_melds = [meld] + other_melds
        
        self.memo[hand_tuple] = (best_dw, best_melds)
        return best_dw, best_melds

    # --- NEW MAIN FUNCTION ---
    
    def update_melds_and_deadwood(self):
        """
        Finds the combination of melds that minimizes deadwood.
        This function replaces update_deadwood and update_sets_and_runs.
        """
        
        # Memoization cache for the recursion
        self.memo = {} 
        
        # Use a hashable tuple for memoization
        hand_tuple = tuple(sorted(self.hand, key=lambda x: (x.suit, int(x.rank))))
        
        best_dw, best_melds = self._find_best_meld_combination(hand_tuple)
        
        self.deadwood = best_dw
        
        # Flatten melds for display/condition checking
        self.sets = []
        self.runs = []
        
        for meld_tuple in best_melds:
            meld = list(meld_tuple)
            # Classify as set or run
            is_set = True
            for i in range(len(meld) - 1):
                if meld[i].rank != meld[i+1].rank:
                    is_set = False
                    break
            if is_set:
                self.sets.extend(meld)
            else:
                self.runs.extend(meld)

    def deal_hand(self, game):
        """
        Deals and sorts the hand of a player

        :return: sorted hand
        """
        hand = []
        for card in range(1, 11):
            card = game.random_choice()
            if card:
                hand.append(card)

        return self.sort_hand(hand)

    def reset_player(self, game, player):
        """
        Resets the relevant attributes of the player for starting a new round

        :param game: game object
        :param player: player object
        """
        self.selected_card = None  # stores card and position of card
        self.swap = None
        self.deadwood = 0
        self.hand = game.new_hands[player.pid] # This seems to be from the old network code
        if not self.hand:
            self.hand = self.deal_hand(game) # Fallback
        self.sets = []
        self.runs = []
        self.ready = False
        self.score = game.p_scores[player.pid]

    @staticmethod
    def sort_hand(hand):
        """
        Sorts the given players hand by suit, then rank
        :return: sorted hand (list)
        """
        return sorted(hand, key=lambda x: (x.suit, int(x.rank)), reverse=False)