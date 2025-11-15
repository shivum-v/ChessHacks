from .utils import chess_manager, GameContext
from chess import Move
import chess
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .model import ImprovedAlphaZeroNet

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
uci_to_idx = {}
idx_to_uci = []

def build_move_index():
    """Build move index mapping (same as in training)"""
    global uci_to_idx, idx_to_uci
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

def move_to_index(move: chess.Move):
    """Convert move to index"""
    return uci_to_idx.get(move.uci(), 0)

def encode_board(board: chess.Board):
    """Enhanced board encoding (20 channels) - same as training"""
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
    
    # Repetition indicator similar to training (0, 0.5, 1.0)
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

class Node:
    """MCTS Node"""
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior

def mcts_search(board, simulations=100, c_puct=2.0):
    """MCTS search to find best move"""
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
            
            # Expand
            legal_moves = list(sim_board.legal_moves)
            legal_priors = []
            move_list = []
            
            for mv in legal_moves:
                idx = move_to_index(mv)
                legal_priors.append(probs[idx])
                move_list.append(mv)
            
            total = sum(legal_priors) + 1e-8
            legal_priors = [p / total for p in legal_priors]
            
            for mv, p in zip(move_list, legal_priors):
                node.children[mv] = Node(parent=node, prior=p)
        
        # Backpropagate
        for node in reversed(search_path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    # Return move with highest visit count and probabilities
    if not root.children:
        return None, {}
    
    moves, nodes = zip(*root.children.items())
    visits = np.array([n.N for n in nodes])
    
    # Create probability distribution for logging
    probs = visits / visits.sum()
    move_probs = {move: float(prob) for move, prob in zip(moves, probs)}
    
    best_move = moves[int(np.argmax(visits))]
    return best_move, move_probs

# Initialize model and move index on import
build_move_index()
model = ImprovedAlphaZeroNet().to(device)

# Try to load trained weights - check multiple locations
possible_paths = [
    Path(__file__).parent.parent / "trained_model.pt",  # Root directory
    Path(__file__).parent / "trained_model.pt",  # src directory
    Path("/src/trained_model.pt"),  # Modal deployment path
]

model_path = None
for path in possible_paths:
    if path.exists():
        model_path = path
        break

if model_path:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[INFO] Loaded trained model from {model_path}")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("[INFO] Using untrained model")
else:
    print(f"[WARNING] Model file not found in any of: {possible_paths}")
    print("[INFO] Using untrained model - train the model first using nn.ipynb")

@chess_manager.entrypoint
def chess_bot(ctx: GameContext):
    """Main entry point - uses neural network with MCTS"""
    print(f"[MOVE {len(ctx.board.move_stack) + 1}] Thinking...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Use MCTS to find best move - more simulations for better play
    # Increase simulations based on time left (more time = deeper search)
    base_simulations = 200
    if ctx.timeLeft > 30000:  # More than 30 seconds
        simulations = 400
    elif ctx.timeLeft > 10000:  # More than 10 seconds
        simulations = 300
    else:
        simulations = base_simulations
    
    best_move, move_probs = mcts_search(ctx.board, simulations=simulations, c_puct=2.5)
    
    if best_move is None:
        # Fallback to random if MCTS fails
        import random
        best_move = random.choice(legal_moves)
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    
    # Log probabilities
    ctx.logProbabilities(move_probs)
    
    print(f"[MOVE {len(ctx.board.move_stack) + 1}] Selected: {best_move.uci()}")
    return best_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins"""
    print("[INFO] New game started - resetting state")
    pass
