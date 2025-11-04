import pygame
import numpy as np
import time

# Local imports
from game import Game, Deck, Card
from player import Player
from conditions import Conditions
from gin_rummy_wrapper import GinRummyPygameEnv
from agents.ppo_agent import PPOAgent

# --- Pygame Setup and Asset Loading ---
# These assets are loaded *after* pygame is initialized
# inside the play_game function.
CARD_BACK = None
logo_t = None
table = None
dw_border = None
scoreboard = None
player_logo = None
sort_btn_img = None
pass_btn_img = None
win_btn_img = None

def load_assets():
    """Loads all pygame assets."""
    global CARD_BACK, logo_t, table, dw_border, scoreboard, player_logo
    global sort_btn_img, pass_btn_img, win_btn_img
    try:
        CARD_BACK = pygame.transform.scale(pygame.image.load("game/deck_images/card_back.png"), (90, 122))
        logo_t = pygame.transform.scale(pygame.image.load("game/images/logo_t.png"), (200, 120))
        table = pygame.transform.scale(pygame.image.load("game/images/card_table.png"), (1200, 800))
        dw_border = pygame.transform.scale(pygame.image.load("game/images/deadwood_border.png"), (600, 200))
        scoreboard = pygame.transform.scale(pygame.image.load("game/images/scoreboard.png"), (600, 180))
        player_logo = pygame.transform.scale(pygame.image.load("game/images/player_2.png"), (170, 180))
        sort_btn_img = pygame.transform.scale(pygame.image.load("game/images/sort_btn.png"), (120, 50))
        pass_btn_img = pygame.transform.scale(pygame.image.load("game/images/pass_btn.png"), (120, 120))
        win_btn_img = pygame.transform.scale(pygame.image.load("game/images/win_btn.png"), (125, 125))
    except pygame.error as e:
        print(f"Error loading assets: {e}")
        print("Please ensure 'game/deck_images' and 'game/images' folders are present.")
        pygame.quit()
        exit()

# --- Button Rects ---
# Define button rectangles for collision detection
DECK_RECT = pygame.Rect(520, 401, 90, 122)
DISCARD_RECT = pygame.Rect(425, 410, 70, 100)
SORT_RECT = pygame.Rect(600, 665, 120, 50)
WIN_BTN_RECT = pygame.Rect(270, 400, 125, 125)
PASS_BTN_RECT = pygame.Rect(640, 400, 120, 120)


# --- Drawing Functions (from your original file) ---

def game_screen(win, game, player):
    font = pygame.font.SysFont("comicsans", 30)
    
    # Simplified score display
    p1_score = font.render(f"{int(game.p_scores[0])}", True, "white")
    p2_score = font.render(f"{int(game.p_scores[1])}", True, "white")
    
    win.blit(table, (-100, 80))
    win.blit(scoreboard, (180, 60))
    win.blit(dw_border, (200, 600))
    win.blit(sort_btn_img, SORT_RECT.topleft)
    win.blit(logo_t, (405, 410))
    
    win.blit(player_logo, (420, 220))
    win.blit(CARD_BACK, DECK_RECT.topleft)
    score = font.render("/10 Deadwood", True, "white")
    win.blit(score, (320, 675))
    
    # Update deadwood before drawing
    player.update_melds_and_deadwood()
    dw_score = font.render(str(player.deadwood), True, "white")
    win.blit(dw_score, (290, 675))
    
    cards_left = font.render(f"{len(game.cards)}", True, "white")
    win.blit(cards_left, (523, 403))

    if player.pid == 0:
        win.blit(p1_score, (290, 160))
        win.blit(p2_score, (620, 160))
    else:
        win.blit(p1_score, (620, 160))
        win.blit(p2_score, (290, 160))

    if game.middle_card is not None:
        win.blit(game.get_image(game.middle_card), DISCARD_RECT.topleft)

def display_hand(win, game, hand, x_axis, y_axis, space_between_cards):
    for card in hand:
        card_image = game.get_image(card)
        win.blit(card_image, (x_axis, y_axis))
        x_axis += space_between_cards

def display_pass_btn(win):
    font = pygame.font.SysFont("helvetica", 26)
    pass_text = font.render("Draw", True, "yellow") # Changed from "Pass"
    win.blit(pass_btn_img, PASS_BTN_RECT.topleft)
    win.blit(pass_text, (672, 440))

def display_win_btn(win, player):
    c = Conditions(player.game, player) # Get fresh conditions

    if c.in_hand_11:
        win_text = ""

        # --- SWAPPED LOGIC ---
        # Check for Knock first
        if c.knock_condition:
            win_text = "Knock!"

        # THEN check for Gin. This will overwrite "Knock!" if it's a Gin hand.
        if c.gin_condition:
            win_text = "Gin!"
        # --- END OF FIX ---

        if win_text:
            font = pygame.font.SysFont("None", 40)
            text_render = font.render(win_text, True, "#FFD700")
            win.blit(win_btn_img, WIN_BTN_RECT.topleft)
            
            # Center text
            text_rect = text_render.get_rect(center=WIN_BTN_RECT.center)
            win.blit(text_render, text_rect)

def highlight_main_sets(win, player, x_axis, y_axis, space_between_cards):
    s = pygame.Surface((70, 100), pygame.SRCALPHA)
    s.fill((46, 204, 113, 128))
    
    player.update_melds_and_deadwood() # Ensure sets are up to date
    sets_and_runs = player.sets + player.runs

    for card in player.hand:
        pos = s.get_rect(topleft=(x_axis, y_axis))
        if card in sets_and_runs:
            x, y = pos.x, pos.y
            win.blit(s, (x, y))
        x_axis += space_between_cards

def draw_rect(win, player):
    if player.selected_card:
        x, y = player.selected_card[1][0], player.selected_card[1][1]
        pygame.draw.rect(win, (0, 0, 0), (x, y, 70, 100), 3)

def draw_end_screen(win, game, human_player, ai_player):
    """Draws the end-of-round screen, similar to the user's image."""
    win.fill((10, 20, 40)) # Dark blue background
    
    # --- 1. DEFINE FONTS AND COLORS ---
    font_l = pygame.font.SysFont("comicsans", 40)
    font_m = pygame.font.SysFont("comicsans", 30)
    result_font = pygame.font.SysFont("comicsans", 70, bold=True) 
    win_color = (0, 255, 0) # Green
    lose_color = (255, 0, 0) # Red
    draw_color = (255, 255, 255) # White

    # Ensure both players' melds are final
    human_player.update_melds_and_deadwood()
    ai_player.update_melds_and_deadwood()
    
    # Identify players
    p1_obj = human_player if human_player.pid == 0 else ai_player
    p2_obj = ai_player if human_player.pid == 0 else human_player
    
    p1_label = "You" if p1_obj == human_player else "AI"
    p2_label = "You" if p2_obj == human_player else "AI"
    
    # Find unmelded (deadwood) cards
    p1_melded_cards = set(p1_obj.sets + p1_obj.runs)
    p1_deadwood_cards = [c for c in p1_obj.hand if c not in p1_melded_cards]
    p2_melded_cards = set(p2_obj.sets + p2_obj.runs)
    p2_deadwood_cards = [c for c in p2_obj.hand if c not in p2_melded_cards]

    # Draw Titles (Using new labels)
    win.blit(font_l.render(p1_label, True, "white"), (150, 50))
    win.blit(font_l.render(p2_label, True, "white"), (650, 50))
    
    win.blit(font_m.render("Melds", True, "cyan"), (100, 150))
    win.blit(font_m.render("Melds", True, "cyan"), (550, 150))
    
    # ... (The card drawing loops are all correct, no change here) ...
    # Draw Melds (smaller cards)
    meld_x = 100
    for card in sorted(p1_obj.sets + p1_obj.runs, key=lambda x: (x.suit, int(x.rank))):
        img = pygame.transform.scale(game.get_image(card), (50, 71))
        win.blit(img, (meld_x, 200))
        meld_x += 30
    
    meld_x = 550
    for card in sorted(p2_obj.sets + p2_obj.runs, key=lambda x: (x.suit, int(x.rank))):
        img = pygame.transform.scale(game.get_image(card), (50, 71))
        win.blit(img, (meld_x, 200))
        meld_x += 30

    win.blit(font_m.render("Deadwood", True, "cyan"), (100, 350))
    win.blit(font_m.render("Deadwood", True, "cyan"), (550, 350))

    # Draw Deadwood Cards
    dw_x = 100
    for card in p1_deadwood_cards:
        img = pygame.transform.scale(game.get_image(card), (50, 71))
        win.blit(img, (dw_x, 400))
        dw_x += 30
        
    dw_x = 550
    for card in p2_deadwood_cards:
        img = pygame.transform.scale(game.get_image(card), (50, 71))
        win.blit(img, (dw_x, 400))
        dw_x += 30
    # ... (End of card drawing) ...

    # Draw Scores
    win.blit(font_m.render(f"Deadwood Score: {p1_obj.deadwood}", True, "white"), (100, 500))
    win.blit(font_m.render(f"Deadwood Score: {p2_obj.deadwood}", True, "white"), (550, 500))
    
    # Calculate round score
    p1_round_score = 0
    p2_round_score = 0
    
    winner_pid = game.win_round["player"]
    win_type = game.win_round["type"]
    
    # --- THIS IS THE FIX ---
    # Check for Dead Hand FIRST
    if win_type == "dead_hand":
        p1_round_score = 0 
        p2_round_score = 0
    elif winner_pid == p1_obj.pid: # Player 1 won
        p1_round_score = p2_obj.deadwood - p1_obj.deadwood
        if win_type == "gin": p1_round_score += 50
        if p1_round_score <= 0: # Undercut
            p2_round_score = abs(p1_round_score) + 25 
            p1_round_score = 0
    elif winner_pid == p2_obj.pid: # Player 2 won
        p2_round_score = p1_obj.deadwood - p2_obj.deadwood
        if win_type == "gin": p2_round_score += 50
        if p2_round_score <= 0: # Undercut
            p1_round_score = abs(p2_round_score) + 25
            p2_round_score = 0
    # --- END OF FIX ---

    # --- 3. DETERMINE (Won/Lose/Draw) MESSAGE ---
    result_message = ""
    result_color = draw_color # Default to draw
    is_human_p1 = (p1_label == "You")

    if p1_round_score > p2_round_score: # P1 wins
        result_message = "You Won!" if is_human_p1 else "You Lose!"
        result_color = win_color if is_human_p1 else lose_color
    elif p2_round_score > p1_round_score: # P2 wins
        result_message = "You Lose!" if is_human_p1 else "You Won!"
        result_color = lose_color if is_human_p1 else win_color
    else: # Draw
        result_message = "Draw!"
        result_color = draw_color

    # --- 4. DRAW THE NEW CENTRAL MESSAGE ---
    result_render = result_font.render(result_message, True, result_color)
    WIN_WIDTH = pygame.display.get_surface().get_width()
    result_rect = result_render.get_rect(center=(WIN_WIDTH / 2, 95)) 
    win.blit(result_render, result_rect)

    # Draw Round Scores
    win.blit(font_l.render(f"Round Score: {p1_round_score}", True, "yellow"), (100, 580))
    win.blit(font_l.render(f"Round Score: {p2_round_score}", True, "yellow"), (550, 580))
    
    win.blit(font_m.render("Click to Continue", True, "white"), (400, 700))

def swap_cards(player):
    if player.selected_card and player.swap:
        try:
            a, b = player.hand.index(player.selected_card[0]), player.hand.index(player.swap[0])
            player.hand[a], player.hand[b] = player.hand[b], player.hand[a]
            player.selected_card, player.swap = None, None
        except ValueError:
            player.selected_card, player.swap = None, None


def allow_selection(player, x, y):
    counter = 100
    for card in player.hand:
        surface = pygame.Surface((70, 100))
        pos = surface.get_rect(topleft=(counter, 550))

        if player.selected_card is None and pos.collidepoint(x, y):
            player.selected_card = [card, (pos.x, pos.y)]
            return # Only select one
        elif player.selected_card and pos.collidepoint(x, y):
            # If same card clicked, deselect
            if player.selected_card[0] == card:
                 player.selected_card = None
                 return
            # Otherwise, set as swap
            player.swap = [card, (pos.x, pos.y)]
            return

        counter += 75

# --- New Main Game Loop ---

def update_window_local(win, game, human_player_obj, ai_player_obj, termination):
    """
    Simplified update function. Only draws the state.
    """
    win.fill((70, 126, 235))
    
    if termination:
        # Show new end screen
        draw_end_screen(win, game, human_player_obj, ai_player_obj)
    else:
        # Draw game state
        game_screen(win, game, human_player_obj)
        display_hand(win, game, human_player_obj.hand, 100, 550, 75)
        highlight_main_sets(win, human_player_obj, 100, 550, 75)

        if human_player_obj.selected_card:
            draw_rect(win, human_player_obj)
            swap_cards(human_player_obj) # Allow card swapping

        # Show buttons based on state
        c = Conditions(game, human_player_obj)
        if game.active_p == human_player_obj.pid:
            if c.in_hand_10 and c.turn == 0:
                display_pass_btn(win) # "Draw" button
            elif c.in_hand_11:
                display_win_btn(win, human_player_obj)

    pygame.display.update()


def get_human_action(player, game, x, y, card_to_pz_index):
    """
    Checks mouse click against buttons and returns a valid
    PettingZoo action index, or None.
    """
    # We get the conditions ONCE. This updates all deadwood scores.
    c = Conditions(game, player)
    
    # --- Check Card Selection (Discard) ---
    if c.in_hand_11 and player.selected_card:
        if DISCARD_RECT.collidepoint(x, y):
            card = player.selected_card[0]
            if card in card_to_pz_index:
                action = 6 + card_to_pz_index[card]
                player.selected_card = None 
                return action
                
    # --- Check Button Clicks ---
    
    # Action: Sort
    if SORT_RECT.collidepoint(x, y):
        player.hand = player.sort_hand(player.hand)
        return None 

    # Action: Draw from Deck (PZ Action 2)
    if c.in_hand_10 and DECK_RECT.collidepoint(x, y):
        return 2
    
    # Action: Draw from Deck (Turn 0 "Pass" button)
    if c.in_hand_10 and c.turn == 0 and PASS_BTN_RECT.collidepoint(x, y):
        return 2
        
    # Action: Pick from Discard (PZ Action 3)
    if c.in_hand_10 and c.turn > 0 and DISCARD_RECT.collidepoint(x, y):
        return 3
        
    # --- START OF NEW WIN BUTTON LOGIC ---
    
    # Action: Win Buttons (Gin, Knock)
    if c.in_hand_11 and WIN_BTN_RECT.collidepoint(x, y):
        
        # We now check c.gin_condition, which is the
        # exact same variable the button uses!
        if c.gin_condition:
            print("Gin condition is True. Registering as GIN (Action 5).")
            player.selected_card = None 
            return 5 # Gin Action
        
        # If not Gin, check for Knock
        if c.knock_condition:
            if not player.selected_card:
                print("Must select a card to discard for Knock!")
                return None
            
            card_to_discard = player.selected_card[0]
            if card_to_discard in card_to_pz_index:
                action = 58 + card_to_pz_index[card_to_discard]
                player.selected_card = None 
                return action
            else:
                print("Error: Selected card not found for knock.")
                return None
    
    # --- END OF NEW WIN BUTTON LOGIC ---
    
    # --- No valid action clicked ---
    allow_selection(player, x, y)
    return None
    

def play_game(human_player_id=0, model_path="path/to/your/model.zip"):
    """
    Main game loop to play Human vs PPO Agent.
    """
    
    # --- PYGAME INIT (FIXES THE CRASH) ---
    pygame.font.init()
    pygame.display.init()
    
    WIDTH = 1000
    HEIGHT = 800
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gin Rummy - PPO Agent Test")
    
    # Load assets *after* display is set
    load_assets()
    # --- END OF INIT ---
    
    # --- Setup ---
    env = GinRummyPygameEnv()
    env.reset()
    
    # Load PPO Agent
    try:
        ppo_agent = PPOAgent(model_path=model_path, env=env)
        print(f"Successfully loaded PPO model from {model_path}")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        print("Please ensure 'stable_baselines3' is installed and model path is correct.")
        pygame.quit()
        return

    ai_player_id = 1 - human_player_id
    ai_agent_name = f"player_{ai_player_id}"
    human_agent_name = f"player_{human_player_id}"
    
    human_player_obj = env.player_0 if human_player_id == 0 else env.player_1
    ai_player_obj = env.player_1 if human_player_id == 0 else env.player_0
    
    run = True
    clock = pygame.time.Clock()
    
    # --- Game Loop ---
    while run:
        clock.tick(30)
        
        game = env.game
        
        agent_turn = env.agent_selection
        obs, reward, termination, truncation, info = env.last()
        
        # --- Draw Window ---
        update_window_local(WIN, game, human_player_obj, ai_player_obj, termination)

        # --- Handle Game Over ---
        if termination or truncation:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    env.reset() # Reset game on click
                    # Give first player 11th card
                    card = env.game.random_choice()
                    if card:
                        env._get_current_player().hand.append(card)
                        env._get_current_player().hand = env._get_current_player().sort_hand(env._get_current_player().hand)
                    break # Exit event loop
            continue # Skip rest of game loop

        # --- Handle Turns ---
        action = None
        
        if agent_turn == ai_agent_name:
            # --- AI's Turn ---

            action_mask = obs['action_mask']
            valid_actions = np.where(action_mask)[0]

            # --- FIX: ADDED THIS GUARD RAIL ---
            if len(valid_actions) == 1 and valid_actions[0] == 4:
                # If the only valid move is to declare a dead hand,
                # force the action without asking the model (which causes NaN).
                print("AI forcing Action 4 (Dead Hand) to prevent crash.")
                action = 4
            # --- END OF FIX ---
            else:

                action, _ = ppo_agent.model.predict(obs, deterministic=True)

                
                
                # Validate action
                if not obs['action_mask'][action]:
                    print(f"AI chose invalid action {action}. Sampling from valid.")
                    valid_actions = np.where(obs['action_mask'])[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        print("Error: No valid actions for AI.")
                        action = 4 # Default to dead hand
                
                print(f"AI ({agent_turn}) chose action: {action}")
                time.sleep(0.5) # Pause to see AI move

        else:
            # --- Human's Turn ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False # <-- CORRECT: DO NOT CALL pygame.quit() HERE
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    action = get_human_action(human_player_obj, game, x, y, env.card_to_pz_index)
                    if action is not None:
                        break # Valid action received
            
            if not run:
                break # Exit while loop if user clicked 'X'

        # --- Step Environment ---
        if action is not None:
            if obs['action_mask'][action]:
                env.step(action)
            else:
                if agent_turn == human_agent_name:
                    print(f"Human chose invalid action: {action}. Ignoring.")
        
    # --- End of while loop ---
    pygame.quit()


if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # Change this path to point to your trained PPO model
    MODEL_PATH = "./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final_knock" 
    
    # This will initialize, run, and quit Pygame cleanly one time.
    play_game(human_player_id=0, model_path=MODEL_PATH)