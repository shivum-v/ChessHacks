import modal
import sys
from pathlib import Path

# Create Modal app
app = modal.App("chesshacks-trainer")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "torchaudio")
    .add_local_file("requirements.txt", "/requirements.txt", copy=True)
    .run_commands("pip install -r /requirements.txt")
    .pip_install("numpy", "tqdm")
    .add_local_dir("src", "/src")
)

# Create a volume to persist the trained model
volume = modal.Volume.from_name("chesshacks-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",  # Use GPU for training
    volumes={"/models": volume},
    timeout=86400,  # 24 hours timeout
)
def train_model():
    """Train the chess neural network model"""
    import chess
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    import sys
    
    # Add src to path
    sys.path.insert(0, "/")
    from src.model import ImprovedAlphaZeroNet
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Build move index first to determine number of actions
    uci_to_idx = {}
    idx_to_uci = []
    
    def build_move_index():
        nonlocal uci_to_idx, idx_to_uci
        all_moves = []
        for a in range(64):
            for b in range(64):
                if a != b:
                    all_moves.append(chess.Move(a, b))
        promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for a in range(64):
            rank = a // 8
            if rank == 6:
                for to_sq in range(56, 64):
                    for p in promos:
                        all_moves.append(chess.Move(a, to_sq, p))
        all_moves = sorted(all_moves, key=lambda m: m.uci())
        idx_to_uci = [m.uci() for m in all_moves]
        uci_to_idx = {m.uci(): i for i, m in enumerate(all_moves)}
        print(f"[INFO] Move index built with {len(idx_to_uci)} actions.")
    
    build_move_index()
    num_actions = len(idx_to_uci)
    
    def move_to_index(move: chess.Move):
        return uci_to_idx.get(move.uci(), 0)
    
    # Reinitialize model with correct number of actions
    model = ImprovedAlphaZeroNet(num_actions=num_actions).to(device)
    print(f"[INFO] Model reinitialized with {num_actions} actions")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Board encoding function
    def encode_board(board: chess.Board):
        planes = np.zeros((20, 8, 8), dtype=np.float32)
        
        # Piece positions
        for sq, piece in board.piece_map().items():
            row = 7 - (sq // 8)
            col = sq % 8
            if piece.color == board.turn:
                channel = piece.piece_type - 1
            else:
                channel = piece.piece_type - 1 + 6
            planes[channel, row, col] = 1.0
        
        # Repetition indicator
        rep_val = 0.0
        try:
            if board.is_repetition(3):
                rep_val = 1.0
            elif board.is_repetition(2):
                rep_val = 0.5
            elif board.can_claim_threefold_repetition():
                rep_val = 1.0
        except Exception:
            rep_val = 0.0
        planes[12, :, :] = rep_val
        
        # En passant
        if board.ep_square is not None:
            row = 7 - (board.ep_square // 8)
            col = board.ep_square % 8
            planes[13, row, col] = 1.0
        
        # Castling rights
        if board.turn == chess.WHITE:
            if board.has_kingside_castling_rights(chess.WHITE):
                planes[14, :, :] = 1.0
            if board.has_queenside_castling_rights(chess.WHITE):
                planes[15, :, :] = 1.0
        else:
            if board.has_kingside_castling_rights(chess.BLACK):
                planes[14, :, :] = 1.0
            if board.has_queenside_castling_rights(chess.BLACK):
                planes[15, :, :] = 1.0
        
        # Side to move
        planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        
        # Move count
        planes[17, :, :] = min(board.fullmove_number / 100.0, 1.0)
        
        # Attacked squares
        for sq in range(64):
            row = 7 - (sq // 8)
            col = sq % 8
            if board.is_attacked_by(board.turn, sq):
                planes[18, row, col] = 1.0
            if board.is_attacked_by(not board.turn, sq):
                planes[19, row, col] = 1.0
        
        return torch.tensor(planes, device=device)
    
    # MCTS Node
    class Node:
        def __init__(self, parent=None, prior=0.0):
            self.parent = parent
            self.children = {}
            self.N = 0
            self.W = 0
            self.Q = 0
            self.P = prior
    
    def add_dirichlet_noise(priors, alpha=0.3, epsilon=0.25):
        noise = np.random.dirichlet([alpha] * len(priors))
        return [(1 - epsilon) * p + epsilon * n for p, n in zip(priors, noise)]
    
    def mcts(board, simulations=100, c_puct=2.0, temperature=1.0, add_noise=False):
        root = Node()
        
        for _ in range(simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]
            
            # Selection
            while node.children and not sim_board.is_game_over():
                move, node = max(
                    node.children.items(),
                    key=lambda kv: kv[1].Q + c_puct * kv[1].P * np.sqrt(node.N) / (1 + kv[1].N)
                )
                sim_board.push(move)
                search_path.append(node)
            
            # Evaluation
            if sim_board.is_game_over():
                result = sim_board.result()
                if result == "1-0":
                    value = 1.0 if sim_board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = 1.0 if sim_board.turn == chess.WHITE else -1.0
                else:
                    value = 0.0
            else:
                state = encode_board(sim_board).unsqueeze(0)
                with torch.no_grad():
                    log_probs, value = model(state)
                
                value = float(value.item())
                probs = F.softmax(log_probs, dim=1).cpu().numpy()[0]
                
                legal_moves = list(sim_board.legal_moves)
                if not legal_moves:
                    break
                
                legal_priors = []
                move_list = []
                
                for mv in legal_moves:
                    idx = move_to_index(mv)
                    if idx < len(probs):
                        legal_priors.append(probs[idx])
                    else:
                        legal_priors.append(0.0)  # Fallback for invalid index
                    move_list.append(mv)
                
                total = sum(legal_priors) + 1e-8
                legal_priors = [p / total for p in legal_priors]
                
                if add_noise:
                    legal_priors = add_dirichlet_noise(legal_priors)
                
                for mv, p in zip(move_list, legal_priors):
                    node.children[mv] = Node(parent=node, prior=p)
            
            # Backpropagate
            for node in reversed(search_path):
                node.N += 1
                node.W += value
                node.Q = node.W / node.N
                value = -value
        
        if not root.children:
            return None
        
        moves, nodes = zip(*root.children.items())
        visits = np.array([n.N for n in nodes])
        
        if temperature == 0:
            best_idx = int(np.argmax(visits))
            return moves[best_idx]
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            chosen_idx = np.random.choice(len(moves), p=probs)
            return moves[chosen_idx]
    
    # Self-play episode
    def self_play_episode(temperature_threshold=15):
        board = chess.Board()
        states, pis, rewards = [], [], []
        move_count = 0
        
        while not board.is_game_over():
            temp = 1.0 if move_count < temperature_threshold else 0.1
            move = mcts(board, simulations=50, c_puct=2.0, temperature=temp, add_noise=True)
            
            if move is None:
                break
            
            states.append(encode_board(board).cpu().numpy())
            
            pi = np.zeros(len(idx_to_uci), dtype=np.float32)
            legal_moves = list(board.legal_moves)
            for mv in legal_moves:
                idx = move_to_index(mv)
                if mv == move:
                    pi[idx] = 1.0
            
            pis.append(pi)
            board.push(move)
            move_count += 1
            
            if move_count > 200:
                break
        
        result = board.result()
        if result == "1-0":
            final_reward = 1
        elif result == "0-1":
            final_reward = -1
        else:
            final_reward = 0
        
        rewards = []
        for i in range(len(states)):
            if i % 2 == 0:
                rewards.append(final_reward)
            else:
                rewards.append(-final_reward)
        
        return np.array(states), np.array(pis), np.array(rewards)
    
    # Replay buffer
    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.capacity = capacity
            self.buffer = []
        
        def push(self, states, pis, rewards):
            for s, p, r in zip(states, pis, rewards):
                if len(self.buffer) >= self.capacity:
                    self.buffer.pop(0)
                self.buffer.append((s, p, r))
        
        def sample(self, batch_size):
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
            batch = [self.buffer[i] for i in indices]
            states, pis, rewards = zip(*batch)
            return np.array(states), np.array(pis), np.array(rewards)
        
        def __len__(self):
            return len(self.buffer)
    
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    epochs = 10
    games_per_epoch = 10
    batch_size = 256
    value_loss_weight = 1.0
    policy_loss_weight = 1.0
    
    print("[INFO] Starting training...")
    print(f"Epochs: {epochs}, Games per epoch: {games_per_epoch}, Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        game_lengths = []
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
        
        # Self-play games
        for game_num in tqdm(range(games_per_epoch), desc=f"Games (Epoch {epoch+1}/{epochs})", leave=False):
            states, pis, rewards = self_play_episode()
            game_lengths.append(len(states))
            
            if len(rewards) > 0 and rewards[0] == 1:
                results['1-0'] += 1
            elif len(rewards) > 0 and rewards[0] == -1:
                results['0-1'] += 1
            else:
                results['1/2-1/2'] += 1
            
            replay_buffer.push(states, pis, rewards)
        
        # Training on batches
        num_batches = 0
        if len(replay_buffer) >= batch_size:
            num_batches = max(1, len(replay_buffer) // batch_size)
            for _ in tqdm(range(num_batches), desc="Batches", leave=False):
                states_batch, pis_batch, rewards_batch = replay_buffer.sample(batch_size)
                states_batch = torch.tensor(states_batch, device=device, dtype=torch.float32)
                pis_batch = torch.tensor(pis_batch, device=device, dtype=torch.float32)
                rewards_batch = torch.tensor(rewards_batch, device=device, dtype=torch.float32).unsqueeze(1)
                
                optimizer.zero_grad()
                log_probs, values = model(states_batch)
                policy_loss = -torch.mean(torch.sum(pis_batch * F.log_softmax(log_probs, dim=1), dim=1))
                value_loss = F.mse_loss(values, rewards_batch)
                loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += loss.item()
        
        scheduler.step()
        
        # Metrics
        avg_policy_loss = epoch_policy_loss / max(1, num_batches)
        avg_value_loss = epoch_value_loss / max(1, num_batches)
        avg_total_loss = epoch_total_loss / max(1, num_batches)
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0
        win_rate = (results['1-0'] / max(1, games_per_epoch)) * 100.0
        draw_rate = (results['1/2-1/2'] / max(1, games_per_epoch)) * 100.0
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch+1}/{epochs} | Loss {avg_total_loss:.4f} (P {avg_policy_loss:.4f} / V {avg_value_loss:.4f}) | "
            f"Win {win_rate:.1f}% Draw {draw_rate:.1f}% | GameLen {avg_game_length:.1f} | LR {current_lr:.6f}"
        )
    
    # Save model
    model_path = "/models/trained_model.pt"
    torch.save(model.state_dict(), model_path)
    volume.commit()
    
    print("\n" + "=" * 60)
    print(f"Training completed! Model saved to {model_path}")
    print("=" * 60)
    
    return model_path

@app.local_entrypoint()
def main():
    """Entry point to run training"""
    result = train_model.remote()
    print(f"\n[SUCCESS] Training finished. Model saved at: {result}")

