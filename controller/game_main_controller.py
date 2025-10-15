# In your other class:
from src.tictactoe_api import TicTacToeEnvAPI
from agents.random_agent import RandomAgent

def display_board_state(obs):
    """Helper function to display the current board state"""
    board = obs['observation']
    display = []
    
    for i in range(9):
        if board[i] == 1:  # Player 1 (X)
            display.append('X')
        elif board[i + 9] == 1:  # Player 2 (O)  
            display.append('O')
        else:
            display.append(str(i))
    
    print("\n Board State:")
    print(f" {display[0]} | {display[1]} | {display[2]} ")
    print("---|---|---")
    print(f" {display[3]} | {display[4]} | {display[5]} ")
    print("---|---|---")
    print(f" {display[6]} | {display[7]} | {display[8]} ")
    print()


if __name__=='__main__':
    # Advanced usage with custom actions
    env = TicTacToeEnvAPI(render_mode="ansi")
    agent1 = RandomAgent(env)
    agent1.set_player('player_1')
    
    agent2 = RandomAgent(env)
    agent2.set_player('player_2')
    
    env.reset(seed=42)
    
    step_count = 0
    agents = {'player_1': agent1, 'player_2': agent2}
    
    print("Starting Tic Tac Toe Game")
    print("Player 1: X")
    print("Player 2: O")
    print("="*30)
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        
        if not (term or trunc):
            display_board_state(obs)
            print(f"\n{agent}'s turn ({'X' if agent == 'player_1' else 'O'})")
            print(f"Available actions: {[i for i, mask in enumerate(obs['action_mask']) if mask == 1]}")
        
        if term or trunc:
            action = None
            print(f"\nGame Over! Final reward for {agent}: {reward}")
            if reward > 0:
                print(f"🏆 {agent} wins!")
            elif reward < 0:
                print(f"❌ {agent} loses!")
            else:
                print("🤝 It's a draw!")
        else:
            current_agent = agents[agent]
            action = current_agent.do_action()
            print(f"{agent} plays position {action}")
        
        env.step(action)
        
        if not (term or trunc):
            input('\nPress Enter to continue...')
        
        step_count += 1
        if step_count >= 20:  # Safety limit (tic tac toe should end much sooner)
            print("Max steps reached!")
            break
    
    env.close()
    print(f"\nCompleted {step_count} steps")