import random
import time
import numpy as np
from Dots_and_Boxes import DotsAndBoxes
from MCTS_Node import MCTSNode

class MCTSPlayer:
    def __init__(self, iterations=1000, exploration=1.4):
        self.iterations = iterations
        self.exploration = exploration
        self.box_bonus = 0.7
        self.chain_weight = 0.3
        self.risk_penalty = 0.4
        
    def choose_move(self, game):
        root = MCTSNode(game)
        
        for _ in range(self.iterations):
            node = root
            # Selection phase
            while not node.untried and node.children:
                node = node.best_child(self.exploration)
            
            # Expansion with safety checks
            if node.untried:
                node = self.safe_expand(node)
            
            # Simulation with validation
            result = self.secure_simulate(node.state, game.current_player)
            node.backpropagate(result)
        
        # Final selection with combined metrics
        if not root.children:
            return random.choice(game.legal_moves())
        
        best_child = max(root.children,
                        key=lambda c: c.visits * (1 - self.risk_penalty * c.risk_score))
        return best_child.action

    def safe_expand(self, node):
        """Expansion with move validation and risk assessment"""
        # Prioritize immediate scoring moves
        scoring_moves = [m for m in node.untried 
                        if self.is_scoring_move(node.state, m)]
        
        # Select move to expand
        if scoring_moves:
            move = random.choice(scoring_moves)
            node.untried.remove(move)
        else:
            # Fallback to safest available move
            move = self.least_risky(node.state, node.untried)
            node.untried.remove(move)

        # Create child node with validated state
        try:
            child_state = node.state.clone()
            child_state.make_move(move)
        except ValueError:
            # Skip invalid moves (shouldn't occur with proper legal moves handling)
            return node
        
        child = MCTSNode(child_state, node)
        child.action = move
        child.risk_score = self.calculate_risk(node.state, move)
        
        node.children.append(child)
        return child

    def calculate_risk(self, original_state, action):
        """Safer risk calculation with move validation"""
        try:
            # Create temp state from original state
            temp = original_state.clone()
            temp.make_move(action)
        except ValueError:
            return float('inf')  # Invalid move, maximum risk
        
        risk = 0
        # Check created 3-edge boxes
        for i in range(temp.board_size):
            for j in range(temp.board_size):
                if not temp.claimed[i,j]:
                    edges = sum([
                        temp.horizontal[i,j],
                        temp.horizontal[i+1,j],
                        temp.vertical[i,j],
                        temp.vertical[i,j+1]
                    ])
                    if edges == 3:
                        risk += 2  # Potential opponent opportunity
        
        # Reduce risk if this move scores points
        if self.is_scoring_move(original_state, action):
            risk = max(0, risk - 3)
        
        # Add penalty for creating long chains
        chain_diff = temp.count_long_chains() - original_state.count_long_chains()
        if chain_diff > 0:
            risk += chain_diff * self.chain_weight
            
        return risk

    def secure_simulate(self, state, original_player):
        """Robust simulation with error handling"""
        temp = state.clone()
        depth = 0
        score = 0
        
        try:
            while depth < 100:
                moves = temp.legal_moves()
                if not moves:
                    break
                
                # Prioritize scoring moves
                scoring_moves = [m for m in moves 
                                if self.is_scoring_move(temp, m)]
                if scoring_moves:
                    move = random.choice(scoring_moves)
                else:
                    move = self.least_risky(temp, moves)
                
                temp.make_move(move)
                depth += 1
                
                # Early termination check
                status = temp.status()
                if status is not None:
                    return 1.0 if status == original_player else -1.0
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0
        
        # Evaluate final state
        score_diff = temp.scores[original_player] - temp.scores[1-original_player]
        return np.tanh(score_diff / 5)  # Normalized score

    def is_scoring_move(self, state, move):
        """Validate if move completes a box"""
        try:
            temp = state.clone()
            before = temp.scores[temp.current_player]
            temp.make_move(move)
            return temp.scores[temp.current_player] > before
        except ValueError:
            return False

    def least_risky(self, state, moves):
        """Find safest move from available options"""
        if not moves:
            return None
        risks = [(m, self.calculate_risk(state, m)) for m in moves]
        min_risk = min(r for _, r in risks)
        return random.choice([m for m, r in risks if r == min_risk])

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


def main():
    game = DotsAndBoxes(4)
    ai = MCTSPlayer()
    
    while game.status() is None:
        print("Current board:")
        display_board(game)
        legal_moves = game.legal_moves()
        print(f"Scores: You (Player 0) = {game.scores[0]}, AI (Player 1) = {game.scores[1]}")
        
        if game.current_player == 0:
            print(f"Available moves: {legal_moves}")
            while True:
                try:
                    move = int(input("Enter your move index: "))
                    if move not in legal_moves:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid move! Choose from available moves.")
        else:
            print("AI is thinking...")
            start_time = time.time()
            move = ai.choose_move(game)
            print(f"AI chose move {move} ({(time.time()-start_time):.1f}s)")
            
        game.make_move(move)
    
    print("\nFinal board:")
    display_board(game)
    result = game.status()
    if result == -1:
        print("Game ended in a draw!")
    else:
        print(f"Player {result} wins!")

if __name__ == "__main__":
    main()