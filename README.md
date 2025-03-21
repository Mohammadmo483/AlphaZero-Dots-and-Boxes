# Dots and Boxes AI with Deep Reinforcement Learning

A sophisticated AI agent for Dots and Boxes that combines Monte Carlo Tree Search (MCTS) with neural networks using AlphaZero-style reinforcement learning.

## Features

- 🎮 **Complete Game Engine**: Full implementation of Dots and Boxes with move validation and board state management
- 🤖 **Advanced AI Players**:
  - MCTS with domain-specific heuristics
  - PUCT (Predictor + UCT) AlphaZero-style player
- 🧠 **Neural Network**: Dual-head architecture for policy and value prediction
- 🔄 **Self-Play Framework**: Automated training through self-play matches
- 📈 **Training Pipeline**: Complete reinforcement learning workflow with experience replay


### Training Mode
Train the AI through self-play:
```bash
python dots_and_boxes.py --mode train
```

### Play Mode
Challenge the AI:
```bash
python dots_and-boxes.py --mode play
```

### Key Controls
- Enter numerical move indices when prompted
- View real-time board state and scores
- Watch AI decision timing and risk analysis

## Code Structure

| Component               | Description                                     |
|-------------------------|-------------------------------------------------|
| `DotsAndBoxes`          | Game engine with move validation and scoring    |
| `MCTSNode/MCTSPlayer`   | Monte Carlo Tree Search implementation          |
| `PUCTNode/PUCTPlayer`   | AlphaZero-style search with neural guidance     |
| `GameNetwork`           | Neural network for policy/value prediction      |
| `SelfPlayWrapper`       | Automated self-play generation                  |
| `TrainingPipeline`      | Neural network training utilities               |

## Training Details

The AI improves through:
- 🧠 Policy iteration with experience replay
- 🤖 Parallel MCTS simulations
- 🔄 Neural network-guided tree search
- ⚖️ Domain knowledge integration (scoring moves, chain detection)


## Future Improvements

- [ ] Web-based GUI interface
- [ ] Distributed self-play generation
- [ ] Alternative network architectures
- [ ] Variable board size support
- [ ] Competitive Elo rating system
