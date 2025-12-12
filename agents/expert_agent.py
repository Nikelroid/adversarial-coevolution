from .agent import Agent
import numpy as np
from hand_scoring import score_gin_rummy_hand

class ExpertAgent(Agent):
    """
    Expert Agent for Gin Rummy that uses heuristic hand scoring to make decisions.
    Acts as a teacher for Imitation Learning.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Mapping from action index to functionality will be derived from the wrapper or environment
        # Typically: 
        # 0, 1: Score (unused)
        # 2: Draw from deck
        # 3: Pick from discard
        # 4: Dead hand (unused)
        # 5: Gin
        # 6-57: Discard card (index - 6)
        # 58-109: Knock (index - 58)

    def do_action(self):
        """
        Decide on the best action based on the current state.
        """
        # Get current observation and mask
        # We need to handle both the raw environment and the wrapper if possible, 
        # but typically this agent will be used where 'last()' returns the standard dict.
        
        if hasattr(self.env, 'last'):
            obs, _, _, _, _ = self.env.last()
            mask = obs["action_mask"]
        else:
             # Fallback or specific implementation for raw env
             obs = self.env.observe(self.player)
             # This might not give us the mask directly if not in wrapper, 
             # but we assume wrapper usage as per ppo_agent.py patterns.
             mask = obs.get("action_mask", np.ones(110))

        # Parse observation
        # Obs is typically 4 planes: [Hand, Top Discard, Discard Pile, Opponent]
        # or similar depending on wrapper.
        # wrapper.py: 
        # Plane 0: Player's Hand
        # Plane 1: Top of Discard Pile
        # Plane 2: Cards in discard pile
        
        obs_array = obs['observation']
        
        # Identify current phase based on mask
        # If Draw (2) or Pick (3) are available: Draw Phase
        # If Discard (6-57), Gin (5), Knock (58-109) are available: Action Phase
        
        can_draw = mask[2] == 1
        can_pick = mask[3] == 1
        
        if can_draw or can_pick:
            return self._decide_draw(obs_array, mask)
        else:
            return self._decide_discard_or_knock(obs_array, mask)

    def _decide_draw(self, obs_array, mask):
        """
        Decide whether to draw from deck or pick from discard.
        """
        # 2 = Draw, 3 = Pick
        if mask[3] == 0: # Can't pick
            return 2
        
        if mask[2] == 0 and mask[3] == 1: # Must pick (unlikely in standard rules unless deck empty?)
            return 3

        # Construct current hand from Plane 0
        current_hand_mask = obs_array[0].flatten() # 52 cards
        
        # Identify top discard card from Plane 1
        top_discard_mask = obs_array[1].flatten()
        top_discard_idx = np.where(top_discard_mask == 1)[0]
        
        if len(top_discard_idx) == 0:
            # No discard card visible/available? Should just draw.
            return 2
            
        top_card_idx = top_discard_idx[0]
        
        # 1. Score hand with current cards
        current_score = score_gin_rummy_hand(current_hand_mask, hand_size=10)
        
        # 2. Score hand IF we picked the discard card
        # We would have 11 cards temporarily. The function expects 10 usually, 
        # but the logic for score_gin_rummy_hand handles finding best melds. 
        # However, to properly evaluate "value of picking", we should simulate 
        # the best discard AFTER picking. 
        # SImpler heuristic: If picking the card increases appropriate potential/score significantly.
        
        # Let's simulate: Pick card, then find best discard.
        new_hand_mask = current_hand_mask.copy()
        new_hand_mask[top_card_idx] = 1 # Add card
        
        best_pick_score = -100
        
        # Iterate over all cards in new hand (11 cards) to discard
        # We can't actually carry 11 cards to the next step, so we predict the score 
        # after we would discard the worst card.
        
        cards_in_hand_11 = np.where(new_hand_mask == 1)[0]
        
        for discard_idx in cards_in_hand_11:
            # Cannot discard the card we just picked in some rule variations (or it's just bad play)
            # But standard gin allows it (though it's a pass).
            # The wrapper logic 6-57 means discarding card at index i.
            
            # Temporary hand of 10
            temp_mask = new_hand_mask.copy()
            temp_mask[discard_idx] = 0
            
            score = score_gin_rummy_hand(temp_mask, hand_size=10)
            if score > best_pick_score:
                best_pick_score = score
                
        # Heuristic: If best potential score after picking is better than current score + threshold
        # (Meaning the picked card helped improve the hand)
        # Current score is "score of 10 cards".
        # We need to compare "Score after Drawing from Deck" vs "Score after Picking Discard".
        # We don't know what's in the deck. Expected value of deck draw?
        # A simple expert rule: If the discard card creates a new meld or extends one, take it.
        # Otherwise draw.
        
        # If the score improved significantly, take it.
        # Threshold: 0.0 means "strictly better".
        if best_pick_score > current_score + 0.05: 
             return 3 # Pick
        
        return 2 # Draw default

    def _decide_discard_or_knock(self, obs_array, mask):
        """
        Decide whether to Gin, Knock, or Discard.
        """
        # Plane 0 has the 11 cards (after draw)
        current_hand_mask = obs_array[0].flatten()
        cards_in_hand = np.where(current_hand_mask == 1)[0]
        
        # 1. Check for Gin (Action 5)
        if mask[5] == 1:
            # We can Gin. Is it always best to Gin? Yes.
            return 5
            
        # 2. Check for Knock (Actions 58-109)
        # We can knock if we discard a specific card and deadwood is low enough.
        # But maybe we want to play for Gin?
        # Safe Expert: If can knock, compare score.
        
        # Let's assess the best move among Discard and Knock.
        
        best_action = -1
        best_score = -100
        
        # Iterate through possible discards (Action 6 to 57)
        # Action i correpsonds to card (i-6)
        
        possible_discards = []
        
        # Standard discards
        for i in range(6, 58):
            if mask[i] == 1:
                card_idx = i - 6
                # Simulate hand after discard
                temp_mask = current_hand_mask.copy()
                temp_mask[card_idx] = 0
                score = score_gin_rummy_hand(temp_mask, hand_size=10)
                
                if score > best_score:
                    best_score = score
                    best_action = i
        
        # Knocks
        # Actions 58-109. Action i corresponds to card (i-58)
        # If we knock, the hand ends. We should prefer knocking if we have a good win chance.
        # In this scoring system, a higher score means better melds/lower deadwood.
        # If we can knock, it usually means we have a computed score.
        
        for i in range(58, 110):
            if mask[i] == 1:
                card_idx = i - 58
                # Simulate hand after discard
                temp_mask = current_hand_mask.copy()
                temp_mask[card_idx] = 0
                score = score_gin_rummy_hand(temp_mask, hand_size=10)
                
                # Check deadwood for knowing if we SHOULD knock.
                # score_gin_rummy_hand returns a normalized score. Not raw deadwood.
                # We might need raw deadwood to decide if knocking is safe vs opponent.
                # But as a simple expert, if we can knock, we usually should unless we are greedy for Gin.
                
                # Bonus for knocking (ending game with win)
                knock_bonus = 0.5 
                
                if score + knock_bonus > best_score:
                    best_score = score + knock_bonus
                    best_action = i
        
        if best_action == -1:
            # Fallback: check for any valid action (e.g. Dead Hand or just failed logic)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                best_action = valid[0] # Just take first valid
            else:
                pass # Should not happen if game is not done
                    
        return best_action

    def set_player(self, player):
        self.player = player
