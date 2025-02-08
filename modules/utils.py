import chess
import torch
import yaml
from typing import Dict


piece_to_index: Dict[str, int] = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
}


# Convert from chess.Board to our representation
def board_to_rep(board: chess.Board) -> torch.Tensor:
    features = torch.zeros(768, dtype=torch.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_index = square
            piece_offset = piece_to_index[piece.symbol()]
            piece_type = piece_offset % 6
            color_offset = piece_offset // 6
            feature_index = square_index * 12 + color_offset * 6 + piece_type
            features[feature_index] = 1

    return features


# Convert from FEN to our representation
def fen_to_rep(fen: str) -> torch.Tensor:
    fen = fen.split()[0]
    features = torch.zeros(768, dtype=torch.float32)

    # We are starting from A1 to H8. Similar to how it is done in the chess-python package. 
    # FEN on the other hand, starts from A8 to H1.
    rank = 7
    file = 0

    for char in fen:
        if char.isdigit():
            file += int(char)
        elif char == '/':
            rank -= 1
            file = 0
        else:
            square_index = rank * 8 + file
            piece_offset = piece_to_index[char]
            piece_type = piece_offset % 6
            color_offset = piece_offset // 6
            feature_index = square_index * 12 + color_offset * 6 + piece_type
            features[feature_index] = 1
            file += 1

    return features


def load_config(config_path: str) -> Dict:
    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)

    return config
