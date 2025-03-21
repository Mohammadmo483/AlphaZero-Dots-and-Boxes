import torch
import time
from Dots_and_Boxes import DotsAndBoxes
from PUCT_Node import PUCTNode
from Game_Network import GameNetwork

class PUCTPlayer:
    def __init__(self, model, cpuct=1.5, num_simulations=800, use_heuristics=True):
        self.model = model
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        self.use_heuristics = use_heuristics
        self.root = None
    def choose_move(self, game):
        root = PUCTNode(game)
        self.root = PUCTNode(game)
        for _ in range(self.num_simulations):
            node = root
            path = []
            
            # Selection
            while node.children:
                node = node.best_child(self.cpuct)
                path.append(node)
            
            # Expansion
            if node.untried_moves and node.state.status() is None:
                policy, value = self._evaluate(node.state)
                if self.use_heuristics:
                    policy = self._apply_heuristics(node.state, policy)
                node.expand(policy)
                path.append(node)  # Will select child in next iteration
            
            # Evaluation
            if node.state.status() is not None:
                value = self._terminal_value(node.state)
            else:
                _, value = self._evaluate(node.state)
            
            # Backpropagation
            for n in reversed(path):
                n.backpropagate(value)
        
        # Select move with highest visit count
        return max(root.children, key=lambda c: c.N).action

    def _evaluate(self, state):
        """Neural network evaluation with heuristic augmentation"""
        # Convert state to tensor
        encoded = torch.tensor(state.encode(), dtype=torch.float32).unsqueeze(0)
        
        # Get neural network predictions
        with torch.no_grad():
            policy_logits, value = self.model(encoded)
        
        # Convert to probability distribution
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        
        # Filter legal moves and normalize
        legal_moves = state.legal_moves()
        legal_policy = {m: policy[m] for m in legal_moves}
        total = sum(legal_policy.values())
        
        if total == 0:
            # Fallback uniform distribution
            return {m: 1/len(legal_moves) for m in legal_moves}, value.item()
        else:
            legal_policy = {m: p/total for m, p in legal_policy.items()}
            return legal_policy, value.item()

    def _apply_heuristics(self, state, policy):
        """Adjust policy using domain knowledge"""
        adjusted = policy.copy()
        legal_moves = state.legal_moves()
        
        # 1. Scoring move bonus
        scoring_moves = [m for m in legal_moves if self._is_scoring_move(state, m)]
        for m in scoring_moves:
            adjusted[m] *= 2.0
            
        # 2. Risk penalty
        for m in legal_moves:
            risk = self._calculate_risk(state, m)
            adjusted[m] *= max(0.2, 1.0 - risk*0.3)  # Reduce up to 80% for high risk
            
        # 3. Endgame chain strategy
        if state.count_long_chains() > 0:
            chain_parity = state.count_long_chains() % 2
            favorable_moves = []
            for m in legal_moves:
                temp = state.clone()
                temp.make_move(m)
                if temp.count_long_chains() % 2 != chain_parity:
                    favorable_moves.append(m)
            for m in favorable_moves:
                adjusted[m] *= 1.5
                
        # Normalize adjusted policy
        total = sum(adjusted.values())
        return {m: p/total for m, p in adjusted.items()}

    def _is_scoring_move(self, state, action):
        temp = state.clone()
        before = temp.scores[temp.current_player]
        try:
            temp.make_move(action)
            return temp.scores[temp.current_player] > before
        except:
            return False

    def _calculate_risk(self, state, action):
        temp = state.clone()
        try:
            temp.make_move(action)
        except:
            return float('inf')
            
        risk = 0
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
                        risk += 1
        return risk

    def _terminal_value(self, state):
        result = state.status()
        if result == -1:
            return 0.0
        return 1.0 if result == state.current_player else -1.0
    
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
        board_size = 4  # Supported by the network architecture
        game = DotsAndBoxes(board_size)

        # Initialize network and AI player
        network = GameNetwork(input_dim=69, action_dim=40)
        puct_player = PUCTPlayer(
            model=network,
            cpuct=1.5,               # Exploration coefficient
            num_simulations=1000,     # Number of PUCT simulations
            use_heuristics=True      # Combine NN with domain knowledge
        )

        # Load pre-trained weights if available
        # network.load_weights("dots_and_boxes_weights.pth")

        while game.status() is None:
            print("\n" + "="*40)
            display_board(game)
            print(f"\nScores: You (Player 0) = {game.scores[0]}, AI (Player 1) = {game.scores[1]}")

            if game.current_player == 0:
                # Human player's turn
                legal_moves = game.legal_moves()
                print(f"Available moves ({len(legal_moves)}): {sorted(legal_moves)}")

                while True:
                    try:
                        move = int(input("Your move (index): "))
                        if move not in legal_moves:
                            raise ValueError("Invalid move index")
                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}. Choose from available moves.")

                game.make_move(move)
            else:
                # AI's turn with timing and move explanation
                print("\nAI is thinking...")
                start_time = time.time()

                # Get AI move with verbose output
                move = puct_player.choose_move(game)
                think_time = time.time() - start_time

                # Analyze move characteristics
                is_scoring = " (SCORING)" if puct_player._is_scoring_move(game, move) else ""
                risk_level = puct_player._calculate_risk(game, move)

                print(f"AI plays move {move}{is_scoring}")
                print(f"Decision time: {think_time:.1f}s | Estimated risk: {risk_level}/10")

                game.make_move(move)

        # Game over display
        print("\n" + "="*40)
        print("FINAL BOARD:")
        display_board(game)

        result = game.status()
        if result == -1:
            print("\nGAME ENDED IN A DRAW!")
        else:
            print(f"\n{'YOU WON! (Player 0)' if result == 0 else 'AI WON! (Player 1)'}")

if __name__ == "__main__":
    main()