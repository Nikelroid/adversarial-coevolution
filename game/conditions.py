class Conditions:
    def __init__(self, game, player):
        player.update_melds_and_deadwood() 

        self.active_player = game.active_p == player.pid
        self.turn = game.turn

        # --- ADD THESE LINES BACK ---
        self.in_hand_10 = len(player.hand) == 10
        self.in_hand_11 = len(player.hand) == 11
        self.card_selected = True if player.selected_card else False
        self.middle_card = True if game.middle_card else False
        self.end_game = True if game.win_round["win"] else False
        # --- END OF ADDED LINES ---

        # PettingZoo knock condition is deadwood <= 10
        self.knock_condition = player.deadwood <= 10

        # Gin condition: deadwood is 0
        self.gin_condition = player.deadwood == 0

        # Super Gin (all 11 cards in melds)
        self.super_condition = self.in_hand_11 and self.gin_condition # <-- This will now work

        # Update gin_condition to include super_gin
        self.gin_condition = self.gin_condition or self.super_condition