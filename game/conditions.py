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
        self.knock_condition = player.deadwood_knock <= 10

        # Gin condition is a 10-card score of 0 (for a normal Gin/Knock)
        self.gin_condition = (player.deadwood_knock == 0)
        
        # Super Gin is an 11-card score of 0 (all 11 cards melded)
        # We check the 11-card score, which is player.deadwood
        self.super_condition = self.in_hand_11 and (player.deadwood == 0)
        
        # The final "Gin" flag is true if it's a normal Gin OR a Super Gin.
        self.gin_condition = self.gin_condition or self.super_condition