import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state.clone()
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried = state.legal_moves()
        self.action = None
        self.risk_score = 0

    def expand(self):
        action = self.untried.pop()
        child_state = self.state.clone()
        child_state.make_move(action)
        child = MCTSNode(child_state, self)
        child.action = action
        child.risk_score = self.calculate_risk(child_state, action)
        self.children.append(child)
        return child

    def calculate_risk(self, state, action):
        risk = 0
        edge_type, row, col = state.decode(action)
        
        if edge_type == 'h':
            boxes = [(row-1, col), (row, col)] if row < state.board_size else [(row-1, col)]
        else:
            boxes = [(row, col-1), (row, col)] if col < state.board_size else [(row, col-1)]

        for i,j in boxes:
            if i < 0 or j < 0 or i >= state.board_size or j >= state.board_size:
                continue
            edges = 0
            edges += state.horizontal[i,j]
            edges += state.horizontal[i+1,j]
            edges += state.vertical[i,j]
            edges += state.vertical[i,j+1]
            if edges == 3:
                risk += 2  # High risk for creating 3-edge box
            elif edges == 2:
                risk += 0.5  # Moderate risk
        return risk

    def best_child(self, c=1.4):
        best = None
        best_score = -float('inf')
        
        for child in self.children:
            exploit = child.wins / child.visits if child.visits else 0
            explore = math.log(self.visits)/child.visits if child.visits else 0
            risk_penalty = 0.3 * child.risk_score  # Penalize risky nodes
            score = exploit + c * math.sqrt(explore) - risk_penalty
            
            if score > best_score:
                best_score = score
                best = child
        return best

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            if self.state.current_player == self.parent.state.current_player:
                # If the parent is the same player, we negate the result
                self.parent.backpropagate(result)
            else:    
                self.parent.backpropagate(-result)