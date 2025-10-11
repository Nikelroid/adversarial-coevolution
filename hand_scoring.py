import numpy as np
from itertools import combinations


def score_gin_rummy_hand(hand_mask, hand_size=10):
    """
    Score a gin rummy hand from -0.5 to 0.5 based on melds, potential melds, and deadwood.
    
    Args:
        hand_mask: 1D array of length 52 with 1s indicating cards in hand
        hand_size: Number of cards in hand (default 10)
    
    Returns:
        float: Score from approximately -0.5 (worst) to 0.5 (best), not clipped
    """
    # Extract card indices from mask
    cards = [i for i, val in enumerate(hand_mask) if val == 1]
    
    if len(cards) != hand_size:
        raise ValueError(f"Expected {hand_size} cards, got {len(cards)}")
    
    # Convert card indices to (suit, rank) tuples
    def idx_to_card(idx):
        suit = idx // 13  # 0=spades, 1=hearts, 2=diamonds, 3=clubs
        rank = idx % 13   # 0=Ace, 1=2, ..., 12=King
        return (suit, rank)
    
    def card_value(rank):
        """Get deadwood value of a card"""
        if rank == 0:  # Ace
            return 1
        elif rank >= 10:  # J, Q, K
            return 10
        else:
            return rank + 1
    
    card_tuples = [idx_to_card(idx) for idx in cards]
    
    # Find all possible sets (3+ cards of same rank)
    def find_sets(cards):
        sets = []
        by_rank = {}
        for i, (suit, rank) in enumerate(cards):
            if rank not in by_rank:
                by_rank[rank] = []
            by_rank[rank].append(i)
        
        for rank, indices in by_rank.items():
            if len(indices) >= 3:
                sets.append(indices)
            elif len(indices) == 4:
                # Can also form a set of 4
                sets.append(indices)
        
        return sets
    
    # Find all possible runs (3+ consecutive cards of same suit)
    def find_runs(cards):
        runs = []
        by_suit = {0: [], 1: [], 2: [], 3: []}
        
        for i, (suit, rank) in enumerate(cards):
            by_suit[suit].append((rank, i))
        
        for suit in range(4):
            suit_cards = sorted(by_suit[suit])
            if len(suit_cards) < 3:
                continue
            
            # Find consecutive sequences
            i = 0
            while i < len(suit_cards):
                run = [suit_cards[i][1]]
                j = i + 1
                
                while j < len(suit_cards) and suit_cards[j][0] == suit_cards[j-1][0] + 1:
                    run.append(suit_cards[j][1])
                    j += 1
                
                if len(run) >= 3:
                    runs.append(run)
                
                i = j if j > i + 1 else i + 1
        
        return runs
    
    # Find best meld combination (greedy approach)
    def find_best_melds(cards):
        sets = find_sets(cards)
        runs = find_runs(cards)
        all_melds = sets + runs
        
        # Greedy selection: pick melds that cover most high-value cards
        used = set()
        selected_melds = []
        
        # Sort melds by total deadwood value (prioritize high-value cards)
        meld_values = []
        for meld in all_melds:
            total_value = sum(card_value(card_tuples[i][1]) for i in meld)
            meld_values.append((total_value, len(meld), meld))
        
        meld_values.sort(reverse=True)
        
        for _, _, meld in meld_values:
            if not any(i in used for i in meld):
                selected_melds.append(meld)
                used.update(meld)
        
        melded_cards = list(used)
        unmelded_cards = [i for i in range(len(cards)) if i not in used]
        
        return selected_melds, melded_cards, unmelded_cards
    
    # Find 2-card potential melds
    def find_potential_melds(unmelded_indices):
        potential = []
        unmelded = [(i, card_tuples[i]) for i in unmelded_indices]
        
        for i, (idx1, card1) in enumerate(unmelded):
            for idx2, card2 in unmelded[i+1:]:
                suit1, rank1 = card1
                suit2, rank2 = card2
                
                # Same rank (potential set)
                if rank1 == rank2:
                    potential.append([idx1, idx2])
                
                # Same suit and adjacent or 1-gap (potential run)
                elif suit1 == suit2 and abs(rank1 - rank2) <= 2:
                    potential.append([idx1, idx2])
        
        return potential
    
    # Calculate best meld combination
    melds, melded_indices, unmelded_indices = find_best_melds(card_tuples)
    
    # Calculate deadwood value
    deadwood_value = sum(card_value(card_tuples[i][1]) for i in unmelded_indices)
    
    # Find potential melds in unmelded cards
    potential_melds = find_potential_melds(unmelded_indices)
    
    # Scoring components
    num_melded = len(melded_indices)
    num_potential = len(potential_melds)
    
    # Score calculation
    # Base score from melded cards: 0.8 per melded card (max 8 points if all 10 melded)
    meld_score = (num_melded / hand_size) * 8.0
    
    # Bonus for complete melds (sets/runs of 3+): 0.5 per meld
    meld_bonus = len(melds) * 0.5
    
    # Penalty for deadwood: -0.05 per deadwood point (max penalty ~5 for 100 points)
    deadwood_penalty = deadwood_value * 0.05
    
    # Small bonus for potential melds: 0.15 per potential pair
    potential_bonus = num_potential * 0.15
    
    # Combine scores
    raw_score = meld_score + meld_bonus + potential_bonus - deadwood_penalty
    
    # Normalize to -0.5 to 0.5 range WITHOUT clipping
    # Theoretical range: worst = -5 (all deadwood, no melds), best = ~9.5 (all melded)
    # This maps: -5 → -0.5, 9.5 → 0.5, 2.25 → 0
    normalized_score = 2 * (raw_score - 2.25) / 14.5
    
    return normalized_score