import numpy as np
from collections import namedtuple
from game import Game, Card
from player import Player
from conditions import Conditions

class GinRummyPygameEnv:
    """
    A wrapper for the Pygame Gin Rummy implementation to mimic
    the PettingZoo gin_rummy_v4 API.
    
    API mimics:
    - reset()
    - step(action)
    - last()
    - observe(agent)
    - agents
    """
    
    def __init__(self):
        self.game = None
        self.player_0 = None
        self.player_1 = None
        
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.agent_selection = None
        
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        
        self.discard_pile = [] # To track full discard
        
        # Build card mapping dictionaries
        self._build_card_mappings()
        self.reset()

    def _build_card_mappings(self):
        self.card_to_pz_index = {}
        self.pz_index_to_card = {}
        
        pz_suits = ['spades', 'hearts', 'diamonds', 'clubs'] # PZ order
        pz_ranks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] # 1=Ace
        
        pz_idx = 0
        for suit in pz_suits:
            for rank in pz_ranks:
                card = Card(rank, suit)
                self.card_to_pz_index[card] = pz_idx
                self.pz_index_to_card[pz_idx] = card
                pz_idx += 1

    def _get_current_player(self):
        return self.player_0 if self.game.active_p == 0 else self.player_1

    def _get_opponent_player(self):
        return self.player_1 if self.game.active_p == 0 else self.player_0

    def _generate_observation(self, agent_id):
        obs = np.zeros((4, 52), dtype=np.int8)
        player = self.player_0 if agent_id == "player_0" else self.player_1
        
        # Plane 0: Player's Hand
        for card in player.hand:
            if card in self.card_to_pz_index:
                obs[0, self.card_to_pz_index[card]] = 1
        
        # Plane 1: Top of Discard Pile
        if self.game.middle_card and self.game.middle_card in self.card_to_pz_index:
            obs[1, self.card_to_pz_index[self.game.middle_card]] = 1
            
        # Plane 2: Cards in discard pile (excluding top)
        for card in self.discard_pile[:-1]: # All but the last
             if card in self.card_to_pz_index:
                obs[2, self.card_to_pz_index[card]] = 1

        # Plane 3: Opponent's known cards (we only know their discards)
        # This is not tracked in the pygame version, so we leave it empty.
        # PettingZoo's env does this, but our pygame logic doesn't.
        # This is a known limitation of this wrapper.
    
        
        return obs

    def _generate_action_mask(self):
        mask = np.zeros(110, dtype=np.int8)
        player = self._get_current_player()
        c = Conditions(self.game, player)

        if c.in_hand_10:
            # Action 2: Draw from deck
            if len(self.game.cards) > 0:
                mask[2] = 1
            # Action 3: Pick from discard
            if c.middle_card and c.turn > 0: # Cannot pick on turn 0
                mask[3] = 1
            # Special case: Turn 0, only draw is allowed (action 2)
            if c.turn == 0:
                mask[3] = 0 # Disallow pick from discard
                
        elif c.in_hand_11:
            # Action 5: Gin
            if c.gin_condition:
                mask[5] = 1
                
            # Actions 6-57: Discard a card
            for card in player.hand:
                if card in self.card_to_pz_index:
                    idx = self.card_to_pz_index[card]
                    mask[6 + idx] = 1
                    
            # Actions 58-109: Knock
            if c.knock_condition and not c.gin_condition:
                for card in player.hand:
                    # You can knock by discarding any card
                    if card in self.card_to_pz_index:
                        idx = self.card_to_pz_index[card]
                        mask[58 + idx] = 1

        if c.in_hand_10 and np.sum(mask) == 0:
            mask[4] = 1

        # Actions 0, 1 (scoring) and 4 (dead hand) are not used in this simplified logic
        return mask

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.game = Game(gid=0)
        self.player_0 = Player(pid=0, game=self.game)
        self.player_1 = Player(pid=1, game=self.game)
        
        # Deal initial hands
        # self.player_0.hand = self.player_0.deal_hand(self.game)
        # self.player_1.hand = self.player_1.deal_hand(self.game)
        
        self.game.ready = True # Set game to "connected"
        self.discard_pile = [self.game.middle_card]
        
        self.agent_selection = self.agents[self.game.active_p]
        
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        
        # Give first player 11th card
        card = self.game.random_choice()
        if card:
            self._get_current_player().hand.append(card)
            
        self._get_current_player().hand = self._get_current_player().sort_hand(self._get_current_player().hand)

    def last(self):
        agent_id = self.agent_selection
        obs_array = self._generate_observation(agent_id)
        action_mask = self._generate_action_mask()
        
        obs = {"observation": obs_array, "action_mask": action_mask}
        reward = self.rewards[agent_id]
        termination = self.terminations[agent_id]
        truncation = self.truncations[agent_id]
        info = self.infos[agent_id]
        
        return obs, reward, termination, truncation, info

    def step(self, action):
        if self.terminations[self.agent_selection]:
            return # Game is over
            
        player = self._get_current_player()
        opponent = self._get_opponent_player()
        
        # Reset rewards
        self.rewards = {a: 0 for a in self.agents}
        termination = False
        
        try:
            if action == 2: # Draw from deck
                card = self.game.random_choice()
                if card:
                    player.hand.append(card)
                player.hand = player.sort_hand(player.hand)
                self.game.next_turn()
            
            elif action == 3: # Pick from discard
                card = self.game.middle_card
                player.hand.append(card)
                player.hand = player.sort_hand(player.hand)
                
                self.discard_pile.pop()
                self.game.middle_card = self.discard_pile[-1] if self.discard_pile else None
                self.game.next_turn()

            elif action == 4: # Declare dead hand
                self._handle_dead_hand(player, opponent)
                termination = True

            elif 6 <= action <= 57: # Discard
                card_to_discard = self.pz_index_to_card[action - 6]
                if card_to_discard in player.hand:
                    player.hand.remove(card_to_discard)
                self.game.middle_card = card_to_discard
                self.discard_pile.append(card_to_discard)
                self.game.active_p = 1 - self.game.active_p # Switch turn
                self.game.next_turn()
            
            elif action == 5: # Gin
                self._handle_win_condition(player, opponent, "gin")
                termination = True

            elif 58 <= action <= 109: # Knock
                card_to_discard = self.pz_index_to_card[action - 58]
                if card_to_discard in player.hand:
                    player.hand.remove(card_to_discard)
                self.game.middle_card = card_to_discard
                self.discard_pile.append(card_to_discard)
                self._handle_win_condition(player, opponent, "knock")
                termination = True
            
            else:
                print(f"Warning: Unknown action {action}")

        except (ValueError, IndexError) as e:
            print(f"Error processing action {action}: {e}")
            # Handle error, e.g., by picking a random valid action
            pass

        # Update player state
        # player.update_sets_and_runs()
        # player.update_deadwood()
        player.update_melds_and_deadwood()
        
        if termination:
            self.terminations = {a: True for a in self.agents}
            self.game.win_round["win"] = True
            self.game.win_round["player"] = player.pid
        else:
            # Move to next agent
            self.agent_selection = self.agents[self.game.active_p]

    def _handle_win_condition(self, player, opponent, win_type):
        player.update_melds_and_deadwood()
        opponent.update_melds_and_deadwood()
        
        player_dw = player.deadwood
        opponent_dw = opponent.deadwood
        
        player_id = self.agents[player.pid]
        opponent_id = self.agents[opponent.pid]
        
        if win_type == "gin":
            self.rewards[player_id] = 1.0 # Gin reward
            self.rewards[opponent_id] = -opponent_dw / 100.0
            self.game.win_round["type"] = "gin"
        
        elif win_type == "knock":
            self.game.win_round["type"] = "knock"
            if player_dw < opponent_dw: # Successful knock
                self.rewards[player_id] = 0.5 # Knock reward
                self.rewards[opponent_id] = -opponent_dw / 100.0
            else: # Undercut
                self.rewards[player_id] = -player_dw / 100.0
                self.rewards[opponent_id] = 0.5 # Opponent gets knock reward

    def _handle_dead_hand(self, player, opponent):
        """Calculates rewards for a dead hand (no valid moves)."""
        player.update_melds_and_deadwood()
        opponent.update_melds_and_deadwood()
        
        player_dw = player.deadwood
        opponent_dw = opponent.deadwood

        player_id = self.agents[player.pid]
        opponent_id = self.agents[opponent.pid]

        # Both players are penalized
        self.rewards[player_id] = -player_dw / 100.0
        self.rewards[opponent_id] = -opponent_dw / 100.0
        self.game.win_round["type"] = "dead_hand"

    def observe(self, agent):
        obs_array = self._generate_observation(agent)
        action_mask = self._generate_action_mask() if agent == self.agent_selection else np.zeros(110, dtype=np.int8)
        return {"observation": obs_array, "action_mask": action_mask}
        
    def render(self):
        # This will be handled by play_ui.py
        pass
        
    def close(self):
        pygame.quit()