import tkinter as tk
from tkinter import font
import numpy as np
import time

# --- Assume these are in the correct paths ---
from src.tictoctoe_api import TicTacToeEnvAPI
from agents.ppo_agent import PPOAgent

MODEL_PATH = './ppo_tictactoe_final'

# --- REVISED AND CORRECTED FUNCTION ---
def normalize_board(obs, current_agent):
    """
    Turn obs['observation'] into a flat length-9 board where:
      1 -> player_1 (X)
     -1 -> player_2 (O)
      0 -> empty
    """
    board_raw = obs['observation']
    arr = np.asarray(board_raw)

    # This is the primary case for the PettingZoo environment's output.
    if arr.shape == (3, 3, 2):
        # The observation is composed of two "planes" or layers.
        # plane 0: A 3x3 grid of the current agent's pieces (1s)
        # plane 1: A 3x3 grid of the other agent's pieces (1s)
        
        current_player_plane = arr[:, :, 0].flatten()
        other_player_plane = arr[:, :, 1].flatten()

        flat_board = np.zeros(9, dtype=int)

        # We must know whose turn it is to correctly assign X and O.
        if current_agent == 'player_1': # Human's turn
            # The current player is X (1), the other is O (-1).
            flat_board[current_player_plane == 1] = 1
            flat_board[other_player_plane == 1] = -1
        else: # AI's turn (player_2)
            # The current player is O (-1), the other is X (1).
            flat_board[current_player_plane == 1] = -1
            flat_board[other_player_plane == 1] = 1
            
        return flat_board
        
    # Other formats can be kept as fallbacks if needed.
    if arr.ndim == 2 and arr.shape == (2, 9):
        p1 = arr[0].astype(int)
        p2 = arr[1].astype(int)
        flat = np.zeros(9, dtype=int)
        flat[p1 == 1] = 1
        flat[p2 == 1] = -1
        return flat
        
    raise ValueError(f"Unrecognized or unhandled observation shape: {arr.shape}")


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.root.geometry("400x550")
        self.root.resizable(False, False)

        # --- Game Logic Initialization ---
        self.env = TicTacToeEnvAPI()
        self.ai_agent = PPOAgent(model_path=MODEL_PATH,env=self.env)
        self.ai_agent.set_player('player_2')
        self.human_player = 'player_1'
        self.ai_player = 'player_2'
        self.game_over = False
        self.last_human_move = None
        self.last_ai_move = None
        self.current_agent = None  # Tracks the agent whose turn it is

        # --- UI Elements ---
        self.buttons = []
        self.button_font = font.Font(family='Helvetica', size=24, weight='bold')
        self.status_font = font.Font(family='Helvetica', size=12)
        self.move_font = font.Font(family='Courier', size=10)

        # Status Label
        self.status_label = tk.Label(root, text="Your turn (X)", font=self.status_font, pady=10)
        self.status_label.pack()

        # Moves Display Frame
        moves_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=1)
        moves_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(moves_frame, text="Moves:", font=self.move_font, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=2)
        
        self.moves_text = tk.Text(moves_frame, height=3, font=self.move_font, state=tk.DISABLED, bg="#f0f0f0")
        self.moves_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Board Frame
        board_frame = tk.Frame(root)
        board_frame.pack()
        
        for i in range(9):
            button = tk.Button(
                board_frame, 
                text="", 
                width=4, 
                height=2, 
                font=self.button_font,
                command=lambda idx=i: self.on_human_move(idx)
            )
            button.grid(row=i//3, column=i%3, padx=2, pady=2)
            self.buttons.append(button)

        # Reset Button
        reset_button = tk.Button(root, text="New Game", command=self.start_new_game)
        reset_button.pack(pady=20)

        # --- Start Game ---
        self.start_new_game()

    def update_moves_display(self):
        """Update the moves text display."""
        self.moves_text.config(state=tk.NORMAL)
        self.moves_text.delete(1.0, tk.END)
        
        moves_log = ""
        if self.last_human_move is not None:
            moves_log += f"You (X): Position {self.last_human_move}\n"
        if self.last_ai_move is not None:
            moves_log += f"AI (O): Position {self.last_ai_move}\n"
        
        self.moves_text.insert(tk.END, moves_log if moves_log else "No moves yet.")
        self.moves_text.config(state=tk.DISABLED)

    def start_new_game(self):
        """Resets the game to its initial state."""
        self.env.reset(seed=int(time.time()))
        self.game_over = False
        self.last_human_move = None
        self.last_ai_move = None
        self.current_agent = self.human_player # Human (player_1) starts
        self.update_board_ui()
        self.update_moves_display()
        self.status_label.config(text="Your turn (X)")

    def update_board_ui(self):
        """Syncs the UI buttons with the current game state."""
        obs, _, _, _, _ = self.env.get_current_state()
        # Use our internal tracker for the current agent
        flat_board = normalize_board(obs, self.current_agent)
        mask = np.asarray(obs["action_mask"]).astype(bool)

        for i in range(9):
            self.buttons[i].config(state=tk.DISABLED, text="")
            if flat_board[i] == 1:
                self.buttons[i].config(text="X", disabledforeground="#3498db")
            elif flat_board[i] == -1:
                self.buttons[i].config(text="O", disabledforeground="#e74c3c")
            elif not self.game_over:
                self.buttons[i].config(state=tk.NORMAL if mask[i] else tk.DISABLED)

    def on_human_move(self, index):
        """Handles the logic when a human player clicks a button."""
        if self.game_over:
            return

        self.last_human_move = index
        self.env.step(index)
        self.current_agent = self.ai_player # Switch turn to AI
        
        self.update_board_ui()
        self.update_moves_display()

        if self.check_game_over():
            return

        self.status_label.config(text="AI is thinking...")
        self.root.after(700, self.on_ai_move)

    def on_ai_move(self):
        """Handles the logic for the AI's turn."""
        if self.game_over:
            return

        action = self.ai_agent.do_action()
        self.last_ai_move = action
        self.env.step(action)
        self.current_agent = self.human_player # Switch turn back to Human
        
        self.update_board_ui()
        self.update_moves_display()

        if self.check_game_over():
            return
            
        self.status_label.config(text="Your turn (X)")

    def check_game_over(self):
        """Checks for a win, loss, or draw and updates the UI."""
        obs, _, term, trunc, _ = self.env.get_current_state()
        if not (term or trunc):
            return False

        self.game_over = True
        # Use our tracker for the final board state normalization
        flat_board = normalize_board(obs, self.current_agent)
        
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        winner = None
        for a,b,c in lines:
            if flat_board[a] == flat_board[b] == flat_board[c] and flat_board[a] != 0:
                winner = 'player_1' if flat_board[a] == 1 else 'player_2'
                break
        
        if winner == self.human_player:
            self.status_label.config(text="🏆 You win!")
        elif winner == self.ai_player:
            self.status_label.config(text="😞 AI wins!")
        else:
            self.status_label.config(text="It's a draw!")
            
        for button in self.buttons:
            button.config(state=tk.DISABLED)

        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

