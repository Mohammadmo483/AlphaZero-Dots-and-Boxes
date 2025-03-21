import math

class PUCTNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state.clone()
        self.parent = parent
        self.children = []
        self.N = 0      # Visit count
        self.Q = 0.0    # Average value
        self.P = prior  # Prior probability
        self.action = None
        self.untried_moves = state.legal_moves()

    def expand(self, policy):
        """Expand node using policy probabilities"""
        for action in self.untried_moves:
            child_state = self.state.clone()
            try:
                child_state.make_move(action)
            except ValueError:
                continue  # Skip invalid moves
                
            child = PUCTNode(child_state, self, policy.get(action, 0.0))
            child.action = action
            self.children.append(child)
        self.untried_moves = []

    def best_child(self, cpuct):
        """Select child using PUCT formula with heuristics"""
        max_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # Base PUCT formula
            exploration = cpuct * child.P * math.sqrt(self.N) / (1 + child.N)
            
            # Heuristic bonus for scoring moves
            heuristic_bonus = 0.0
            if self._is_scoring_move(child.action):
                heuristic_bonus = 0.2  # Additional exploration bonus
                
            # Heuristic penalty for risky moves
            risk_penalty = self._calculate_risk(child.action) * 0.1
                
            score = child.Q + exploration + heuristic_bonus - risk_penalty
            
            if score > max_score:
                max_score = score
                best_child = child
        return best_child

    def update(self, value):
        """Update node statistics"""
        self.N += 1
        self.Q += (value - self.Q) / self.N  # Incremental average

    def backpropagate(self, value):
        self.update(value)
        if self.parent is not None:
            if self.state.current_player == self.parent.state.current_player:

                self.parent.backpropagate(value)
            else:
               
                self.parent.backpropagate(-value)

    def _is_scoring_move(self, action):
        """Check if action completes a box"""
        temp = self.state.clone()
        before = temp.scores[temp.current_player]
        try:
            temp.make_move(action)
            return temp.scores[temp.current_player] > before
        except:
            return False

    def _calculate_risk(self, action):
        """Calculate move risk using existing heuristic"""
        temp = self.state.clone()
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