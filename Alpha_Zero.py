import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Dots_and_Boxes import DotsAndBoxes
from PUCT_Player import PUCTPlayer
from Game_Network import GameNetwork
from MCTS_Player import MCTSPlayer


class AlphaZeroTrainer:
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.network = GameNetwork()
        self.best_network = GameNetwork()
        self.elo = 1200
        self.self_play = SelfPlayWrapper(board_size)
        self.pipeline = TrainingPipeline(self.network)
        
    def train(self, iterations=10, games_per_iter=100):
        for iteration in range(iterations):
            print(f"\n=== Training Iteration {iteration+1}/{iterations} ===")
            
            # Generate self-play data
            print("Generating self-play games...")
            data = self.self_play.generate_game(self.network, games_per_iter)
            
            # Train network
            print("Training network...")
            self.pipeline.train(data)
            
            # Evaluate
            print("Evaluating...")
            current_rating = self.evaluate()
            
            # Save best network
            if current_rating > self.elo:
                self.best_network.load_state_dict(self.network.state_dict())
                self.elo = current_rating
                torch.save(self.network.state_dict(), f"best_model_{iteration}.pth")
            
            print(f"Current ELO: {self.elo:.1f}")
            print(f"Policy Loss: {np.mean(self.pipeline.loss_policy[-100:]):.4f}")
            print(f"Value Loss: {np.mean(self.pipeline.loss_value[-100:]):.4f}")
    
    def evaluate(self, num_games=20):
        wins = 0
        mcts_player = MCTSPlayer(iterations=1000)
        
        for _ in range(num_games):
            game = DotsAndBoxes(self.board_size)
            az_player = PUCTPlayer(self.network, num_simulations=100)
            
            while game.status() is None:
                if game.current_player == 0:
                    move = az_player.choose_move(game)
                else:
                    move = mcts_player.choose_move(game)
                game.make_move(move)
            
            if game.status() == 0:
                wins += 1
                
        win_rate = wins / num_games
        new_elo = self.elo + 32 * (win_rate - 0.5)  # Simple ELO update
        return new_elo

# =============================================================================
class SelfPlayWrapper:
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.data_buffer = deque(maxlen=100000)
        
    def generate_game(self, network, num_games=1):
        for _ in range(num_games):
            game = DotsAndBoxes(self.board_size)
            game_history = []
            
            while game.status() is None:
                player = PUCTPlayer(network, num_simulations=200)  # Increased simulations
                move = player.choose_move(game)
                
                # Get valid policy from root
                policy = np.zeros(40)
                if player.root and player.root.children:
                    total_visits = sum(c.N for c in player.root.children)
                    if total_visits > 0:
                        for child in player.root.children:
                            policy[child.action] = child.N / total_visits
                    else:  # Fallback for no visits
                        policy[list(game.legal_moves())] = 1/len(game.legal_moves())
                else:  # Fallback for no children
                    legal = game.legal_moves()
                    policy[list(legal)] = 1/len(legal)
                
                # Add noise for exploration
                policy = 0.75*policy + 0.25*np.random.dirichlet([0.3]*40)
                policy /= policy.sum()
                
                game_history.append((game.encode(), policy, None))
                game.make_move(move)
            
            # Assign final values
            result = game.status()
            final_value = 0
            if result != -1:
                final_value = 1 if result == 0 else -1
                
            # Update experience with final values
            for i, (s, p, _) in enumerate(game_history):
                player_value = final_value if i%2 == 0 else -final_value
                self.data_buffer.append((s, p, player_value))
                
        return self.data_buffer

# ===========================================================================================
class TrainingPipeline:
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=0.001)
        self.loss_policy = []
        self.loss_value = []
    
    def train(self, data, batch_size=512, epochs=10):
        if not data:
            return  # No data to train on
        
        for epoch in range(epochs):
            # Randomly sample batch from replay buffer
            batch = random.sample(data, min(batch_size, len(data)))
            
            # Unpack and convert to tensors
            states, policies, values = zip(*batch)
            
            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            policies_t = torch.tensor(np.array(policies), dtype=torch.float32)
            values_t = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_logits, value_estimates = self.network(states_t)
            
            # Calculate policy loss using KL divergence
            policy_loss = nn.functional.kl_div(
                input=nn.functional.log_softmax(policy_logits, dim=1),
                target=policies_t,
                reduction='batchmean'
            )
            
            # Calculate value loss using MSE
            value_loss = nn.functional.mse_loss(value_estimates, values_t)
            
            # Combine losses
            total_loss = policy_loss + value_loss + 0.01*torch.norm(policy_logits, p=2)
            
            # Backpropagate
            total_loss.backward()
            self.optimizer.step()
            
            # Store losses for monitoring
            self.loss_policy.append(policy_loss.item())
            self.loss_value.append(value_loss.item())
            
            # Print progress
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Policy Loss: {policy_loss.item():.4f}")
                print(f"Value Loss: {value_loss.item():.4f}")
                print("-------------------")
        
        return np.mean(self.loss_policy[-epochs:]), np.mean(self.loss_value[-epochs:])

# =============================================================================
# Main Execution
# =============================================================================
   
    # Training mode
# python Alpha_Zero.py --mode train

# # Play mode
# python Alpha_Zero.py --mode play
# -----------------------------------------------------------------------------
def display_board(game):
    size = game.board_size
    for row in range(size + 1):
        line = ['o']
        for col in range(size):
            if game.horizontal[row, col]:
                line.append('---o')
            else:
                line.append('   o')
        print(''.join(line))
        
        if row < size:
            line = []
            for col in range(size + 1):
                if game.vertical[row, col]:
                    line.append('|')
                else:
                    line.append(' ')
                if col < size:
                    line.append(' X ' if game.claimed[row,col] else '   ')
            print(''.join(line))

def main(mode='train'):
    if mode == 'train':
        trainer = AlphaZeroTrainer()
        trainer.train(iterations=20, games_per_iter=200)
    else:
        # Human vs AI play
        game = DotsAndBoxes(4)
        network = GameNetwork()
        network.load_state_dict(torch.load("best_model.pth"))
        ai = PUCTPlayer(network, num_simulations=800)
        
        while game.status() is None:
            print("\n" + "="*40)
            display_board(game)
            print(f"Scores: You (Player 0) = {game.scores[0]}, AI (Player 1) = {game.scores[1]}")
            
            if game.current_player == 0:
                legal_moves = game.legal_moves()
                print(f"Available moves: {sorted(legal_moves)}")
                move = int(input("Your move: "))
                game.make_move(move)
            else:
                print("AI thinking...")
                start = time.time()
                move = ai.choose_move(game)
                print(f"AI chose move {move} in {time.time()-start:.1f}s")
                game.make_move(move)
        
        print("\nFinal Result:")
        display_board(game)
        print("You win!" if game.status() == 0 else "AI wins!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'play'], default='play')
    args = parser.parse_args()
    
    main(args.mode)
    