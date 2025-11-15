# /src/main.py
import chess
import torch
import numpy as np
from pathlib import Path

from .utils import chess_manager, GameContext

# ====================================================
# 1. Load your trained AlphaZero-lite neural network
# ====================================================

from .model import AlphaZeroNet

MODEL_PATH = Path(__file__).parent / "trained_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroNet().to(device)

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Loaded trained neural network.")
else:
    print("[WARNING] No model found! Using random untrained weights.")


# ====================================================
# 2. Board encoder used by both training + inference
# ====================================================

def encode_state_from_board(board: chess.Board):
    """
    Convert python-chess board into AlphaZero tensor representation.
    Output shape: (12, 8, 8)
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        channel_offset = 6 if piece.color == chess.WHITE else 0
        channel = channel_offset + (piece.piece_type - 1)
        planes[channel, row, col] = 1.0

    return torch.tensor(planes, device=device)


# ====================================================
# 3. Move indexing utilities (MUST match training)
# ====================================================

uci_to_idx = {}
idx_to_uci = []

def build_move_index():
    global uci_to_idx, idx_to_uci

    all_moves = []

    # all normal from/to
    for a in range(64):
        for b in range(64):
            if a != b:
                all_moves.append(chess.Move(a, b))

    # all promotions
    promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for a in range(64):
        rank = a // 8
        if rank == 6:
            for to_sq in range(56, 64):
                for p in promos:
                    all_moves.append(chess.Move(a, to_sq, promotion=p))

    all_moves = sorted(all_moves, key=lambda m: m.uci())
    idx_to_uci = [m.uci() for m in all_moves]
    uci_to_idx = {uci: i for i, uci in enumerate(idx_to_uci)}

    print(f"[INFO] Move index built with {len(idx_to_uci)} actions.")

build_move_index()

def move_to_index(move: chess.Move):
    return uci_to_idx.get(move.uci(), 0)


# ====================================================
# 4. Minimal MCTS
# ====================================================

class Node:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior


def mcts_select_move(board: chess.Board, simulations=40, c_puct=1.4):
    root = Node()

    for _ in range(simulations):
        node = root
        sim_board = board.copy()

        # --- Selection ---
        while node.children:
            move, node = max(
                node.children.items(),
                key=lambda kv: kv[1].Q + c_puct * kv[1].P *
                (np.sqrt(node.N + 1e-8) / (1 + kv[1].N))
            )
            sim_board.push(move)

        # --- NN Evaluation ---
        state = encode_state_from_board(sim_board).unsqueeze(0)
        with torch.no_grad():
            log_probs, value = model(state)

        probs = log_probs.exp().cpu().numpy()[0]

        # Mask only legal moves
        legal_priors = {}
        pri_sum = 0
        for mv in sim_board.legal_moves:
            idx = move_to_index(mv)
            legal_priors[mv] = probs[idx]
            pri_sum += probs[idx]

        # normalize
        for mv in legal_priors:
            legal_priors[mv] /= (pri_sum + 1e-8)

        # --- Expansion ---
        for mv, prior in legal_priors.items():
            node.children[mv] = Node(parent=node, prior=prior)

        # --- Backprop ---
        v = float(value.item())
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q = cur.W / cur.N
            v = -v
            cur = cur.parent

    # pick move with max visit count
    moves, nodes = tuple(zip(*root.children.items()))
    visits = [n.N for n in nodes]
    best = moves[int(np.argmax(visits))]

    return best, root


# ====================================================
# 5. REQUIRED ENTRYPOINT (their API)
# ====================================================

@chess_manager.entrypoint
def get_move(ctx: GameContext):
    """
    Called every time the engine must produce a move.
    Must return a python-chess Move object.
    """
    board = ctx.board
    print("Engine thinking...")

    best_move, root = mcts_select_move(board, simulations=60)

    # show probabilities in the UI
    probs = {mv: child.N for mv, child in root.children.items()}
    total = sum(probs.values())
    ctx.logProbabilities({mv: n / total for mv, n in probs.items()})

    return best_move


# ====================================================
# 6. OPTIONAL: Reset hook
# ====================================================

@chess_manager.reset
def reset_func(ctx: GameContext):
    print("[INFO] New game started.")
    # clear caches, etc
    pass
