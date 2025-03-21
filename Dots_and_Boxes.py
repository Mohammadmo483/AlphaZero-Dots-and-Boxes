import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque

class DotsAndBoxes:
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.horizontal = np.zeros((board_size + 1, board_size), dtype=bool)
        self.vertical = np.zeros((board_size, board_size + 1), dtype=bool)
        self.claimed = np.zeros((board_size, board_size), dtype=bool)
        self.scores = [0, 0]
        self.current_player = 0
        self.history = []

    def make_move(self, action_index):
        edge_type, row, col = self.decode(action_index)
        if edge_type == 'h' and self.horizontal[row, col]:
            raise ValueError("Invalid horizontal move")
        if edge_type == 'v' and self.vertical[row, col]:
            raise ValueError("Invalid vertical move")

        prev_scores = self.scores.copy()
        prev_player = self.current_player

        if edge_type == 'h':
            self.horizontal[row, col] = True
            boxes = [(row-1, col)] if row > 0 else []
            boxes += [(row, col)] if row < self.board_size else []
        else:
            self.vertical[row, col] = True
            boxes = [(row, col-1)] if col > 0 else []
            boxes += [(row, col)] if col < self.board_size else []

        new_claims = []
        for i, j in boxes:
            if not self.claimed[i,j] and self.horizontal[i,j] and self.horizontal[i+1,j] \
            and self.vertical[i,j] and self.vertical[i,j+1]:
                new_claims.append((i,j))
                self.claimed[i,j] = True

        self.scores[prev_player] += len(new_claims)
        self.current_player = prev_player if new_claims else 1 - prev_player
        self.history.append({
            'action': action_index,
            'claims': new_claims,
            'scores': prev_scores,
            'player': prev_player
        })

    def unmake_move(self):
        if not self.history:
            return
        last = self.history.pop()
        action = last['action']
        edge_type, row, col = self.decode(action)
        
        if edge_type == 'h':
            self.horizontal[row, col] = False
        else:
            self.vertical[row, col] = False
            
        for i,j in last['claims']:
            self.claimed[i,j] = False
            
        self.scores = last['scores'].copy()
        self.current_player = last['player']

    def legal_moves(self):
        moves = []
        h_moves = (self.board_size + 1) * self.board_size
        moves += [r*self.board_size + c 
                 for r,c in np.argwhere(~self.horizontal)]
        moves += [h_moves + r*(self.board_size+1) + c 
                 for r,c in np.argwhere(~self.vertical)]
        return moves

    def decode(self, action):
        h_moves = (self.board_size + 1) * self.board_size
        if action < h_moves:
            return ('h', action//self.board_size, action%self.board_size)
        action -= h_moves
        return ('v', action//(self.board_size+1), action%(self.board_size+1))

    def status(self):
        if not self.legal_moves():
            if self.scores[0] > self.scores[1]:
                return 0
            if self.scores[1] > self.scores[0]:
                return 1
            return -1
        return None

    def clone(self):
        clone = DotsAndBoxes(self.board_size)
        clone.horizontal = self.horizontal.copy()
        clone.vertical = self.vertical.copy()
        clone.claimed = self.claimed.copy()
        clone.scores = self.scores.copy()
        clone.current_player = self.current_player
        clone.history = self.history.copy()
        return clone

    def count_third_edges(self):
        count = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.claimed[i,j]:
                    continue
                edges = 0
                edges += self.horizontal[i,j]
                edges += self.horizontal[i+1,j]
                edges += self.vertical[i,j]
                edges += self.vertical[i,j+1]
                if edges == 3:
                    count += 1
        return count
    
    def encode(self):
        # For board_size=4:
        # horizontal: (5,4)=20, vertical: (4,5)=20, claimed: (4,4)=16, current_player:1, scores:6+6=12; total=69
        h_edges = self.horizontal.flatten().astype(int)
        v_edges = self.vertical.flatten().astype(int)
        claimed = self.claimed.flatten().astype(int)
        current_player = np.array([self.current_player], dtype=int)
        def int_to_bin(x, bits):
            return np.array([(x >> i) & 1 for i in range(bits)])
        score0 = int_to_bin(self.scores[0], 6)
        score1 = int_to_bin(self.scores[1], 6)
        encoded = np.concatenate([h_edges, v_edges, claimed, current_player, score0, score1])
        return encoded.astype(int)
    
    def count_long_chains(self):
        rows, cols = self.board_size, self.board_size
        visited = np.zeros((rows, cols), dtype=bool)
        long_chains = 0
        for i in range(rows):
            for j in range(cols):
                if not self.claimed[i, j] and not visited[i, j]:
                    queue = [(i, j)]
                    visited[i, j] = True
                    component = []
                    while queue:
                        x, y = queue.pop(0)
                        component.append((x, y))
                        # Check neighbors
                        if x > 0 and not self.horizontal[x, y] and not visited[x - 1, y] and not self.claimed[x - 1, y]:
                            visited[x - 1, y] = True
                            queue.append((x - 1, y))
                        if x < rows - 1 and not self.horizontal[x + 1, y] and not visited[x + 1, y] and not self.claimed[x + 1, y]:
                            visited[x + 1, y] = True
                            queue.append((x + 1, y))
                        if y > 0 and not self.vertical[x, y] and not visited[x, y - 1] and not self.claimed[x, y - 1]:
                            visited[x, y - 1] = True
                            queue.append((x, y - 1))
                        if y < cols - 1 and not self.vertical[x, y + 1] and not visited[x, y + 1] and not self.claimed[x, y + 1]:
                            visited[x, y + 1] = True
                            queue.append((x, y + 1))
                    ends = 0
                    for (x, y) in component:
                        neighbors = 0
                        if x > 0 and not self.horizontal[x, y] and ((x - 1, y) in component):
                            neighbors += 1
                        if x < rows - 1 and not self.horizontal[x + 1, y] and ((x + 1, y) in component):
                            neighbors += 1
                        if y > 0 and not self.vertical[x, y] and ((x, y - 1) in component):
                            neighbors += 1
                        if y < cols - 1 and not self.vertical[x, y + 1] and ((x, y + 1) in component):
                            neighbors += 1
                        if neighbors == 1:
                            ends += 1
                    if ends == 2 and len(component) >= 3:
                        long_chains += 1
        return long_chains