# Chess AI Training Guide

## Overview
This chess bot uses a **deep neural network with Monte Carlo Tree Search (MCTS)** to play chess. The model has been significantly improved from the basic version to reduce blunders and play more strategically.

## Major Improvements

### 1. Enhanced Neural Network Architecture
- **Residual Blocks**: 10 residual blocks with batch normalization for better gradient flow
- **Deeper Network**: ~4.5M parameters (vs. ~800K in original)
- **Separate Heads**: Dedicated policy and value heads with proper dimension reduction
- **Better Capacity**: 256 channels throughout residual tower

### 2. Rich Board Encoding (20 Channels)
The model now sees much more of the game state:
- **Piece positions** (12 channels): Own and opponent pieces
- **Repetition counter**: Detect repeated positions (draw detection)
- **En passant square**: Special pawn capture
- **Castling rights**: Both kingside and queenside
- **Side to move**: Who's turn it is
- **Move count**: Game phase indicator
- **Attack maps** (2 channels): Squares attacked by each side

### 3. Improved MCTS
- **Higher simulation count**: 100 simulations per move (vs. 25)
- **Better UCB tuning**: c_puct=2.0 for optimal exploration/exploitation
- **Temperature-based selection**: Exploration in opening, exploitation in endgame
- **Dirichlet noise**: Added during training for better exploration

### 4. Advanced Training
- **Experience Replay**: Buffer of 10,000 positions for stable learning
- **Multiple games per epoch**: 5 games per epoch for diverse experience
- **Learning rate scheduling**: Cosine annealing from 0.001 to 0.00001
- **Gradient clipping**: Max norm of 1.0 to prevent exploding gradients
- **Better loss weighting**: Balanced policy and value losses
- **Regularization**: L2 weight decay (1e-4)

### 5. Comprehensive Metrics
Training now tracks:
- Policy and value losses
- Average game length
- Win/draw rates
- Learning rate progression
- Visual plots saved to `training_metrics.png`

## Training the Model

### Step 1: Install Dependencies
```bash
cd my-chesshacks-bot
pip install -r requirements.txt
```

### Step 2: Open the Training Notebook
```bash
jupyter notebook src/nn.ipynb
# Or use VS Code to open nn.ipynb
```

### Step 3: Run All Cells
Execute the cells in order:
1. **Cell 1**: Install dependencies (if needed)
2. **Cell 2**: Import libraries
3. **Cell 3**: Define improved network architecture
4. **Cell 4**: Enhanced board encoding
5. **Cell 5**: Build move index
6. **Cell 6**: Improved MCTS implementation
7. **Cell 7**: Self-play and replay buffer
8. **Cell 8**: Training loop (this will take time!)
9. **Cell 9**: Visualize training metrics
10. **Cell 10**: Save trained model

### Step 4: Training Parameters
You can adjust these in Cell 8:
```python
epochs = 50              # More epochs = better training (but slower)
games_per_epoch = 5      # More games = more diverse experience
simulations = 50         # MCTS simulations per move
batch_size = 256         # Batch size for training
```

### Expected Training Time
- **Per game**: 5-15 minutes (depends on game length and hardware)
- **Per epoch**: 25-75 minutes (5 games)
- **Full training (50 epochs)**: 20-60 hours

**Recommendation**: Start with 10-20 epochs for initial testing, then increase.

### Hardware Considerations
- **CPU**: Works but slow (~10-15 min per game)
- **GPU (CUDA)**: Much faster (~2-5 min per game)
- **Apple Silicon (MPS)**: Medium speed (~5-8 min per game)

The code automatically detects and uses CUDA if available.

## Using the Trained Model

### Step 1: Ensure Model File Exists
After training, you should have:
```
my-chesshacks-bot/src/trained_model.pt
```

### Step 2: Run the Bot
```bash
cd my-chesshacks-bot
python serve.py
```

The bot will:
1. Load the trained model automatically
2. Use MCTS with neural network guidance for each move
3. Log move probabilities for analysis

### Model Behavior
- **Opening**: Explores more (temperature=1.0 for first 15 moves)
- **Middlegame/Endgame**: More focused (temperature=0.1)
- **MCTS**: 100 simulations per move during actual play
- **Thinking time**: ~10-30 seconds per move (depends on hardware)

## Performance Tips

### For Faster Training
1. **Reduce simulations**: Change `simulations=50` to `25` in self-play
2. **Fewer games**: Use `games_per_epoch=3` instead of 5
3. **Smaller network**: Use `num_res_blocks=5` and `channels=128`

### For Stronger Play
1. **More training**: Increase `epochs` to 100+
2. **More simulations**: Change MCTS to 200+ simulations in `main.py`
3. **Larger network**: Use `num_res_blocks=15` and `channels=512` (requires more memory)

## Architecture Comparison

| Feature | Original | Improved |
|---------|----------|----------|
| Network depth | 3 conv layers | 10 residual blocks |
| Parameters | ~800K | ~4.5M |
| Input features | 12 channels | 20 channels |
| MCTS simulations | 25 | 100 |
| Training epochs | 10 | 50 |
| Experience replay | No | Yes (10K buffer) |
| LR scheduling | No | Yes (cosine) |
| Gradient clipping | No | Yes |
| Batch normalization | No | Yes |

## Troubleshooting

### "IProgress not found" Error
The notebook uses standard `tqdm` (not `tqdm.notebook`), so this shouldn't happen. If it does:
```bash
pip install --upgrade ipywidgets
```

### Out of Memory
Reduce batch size or network size:
```python
batch_size = 128  # Instead of 256
channels = 128    # Instead of 256
```

### Model Not Loading
Make sure `trained_model.pt` is in the `src/` directory:
```bash
ls my-chesshacks-bot/src/trained_model.pt
```

### Slow Training
- Use a GPU if available
- Reduce `simulations` in self-play
- Reduce `games_per_epoch`

## Advanced: Custom Training

### Training Against Specific Openings
Modify `self_play_episode()` to start from specific positions:
```python
def self_play_episode():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    # ... rest of code
```

### Curriculum Learning
Train on progressively harder positions:
1. Start with simple endgames (K+Q vs K)
2. Move to complex endgames
3. Train on full games

### Loading Pre-trained Weights
```python
# In the notebook, before training:
model.load_state_dict(torch.load("previous_model.pt"))
```

## Next Steps

1. **Train for longer**: 50+ epochs for better results
2. **Analyze games**: Look at `training_metrics.png` to see improvement
3. **Test against engines**: Compare against Stockfish or other bots
4. **Iterate**: Adjust hyperparameters based on results

## Key Metrics to Watch

- **Policy Loss**: Should decrease steadily (good predictions)
- **Value Loss**: Should decrease (accurate position evaluation)
- **Avg Game Length**: Should stabilize (consistent play)
- **Win Rate**: Should become more consistent (not random)

Good luck training your chess bot! 🎯♟️
