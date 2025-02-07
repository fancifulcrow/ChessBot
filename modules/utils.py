import chess
import torch
import yaml
from typing import Dict, List

# Piece-square tables for chess pieces
piece_square_tables: Dict[chess.Piece, List[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],

    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],

    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],

    chess.ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    ],

    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],

    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Material values of chess pieces
piece_values: Dict[chess.Piece, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}


piece_to_index: Dict[str, int] = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
}

# Get the position value for a piece on a given square.
def get_position_value(piece: chess.Piece, square: chess.Square, is_endgame: bool) -> int:
    if piece.piece_type == chess.KING and is_endgame:
        table = KING_ENDGAME_TABLE
    else: 
        table = piece_square_tables[piece.piece_type]
    
    rank, file = chess.square_rank(square), chess.square_file(square)
    index = rank * 8 + file if piece.color else 63 - (rank * 8 + file)
    return table[index]


# Check if endgame
def is_endgame(board: chess.Board) -> bool:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minor_pieces = (
        len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
        len(board.pieces(chess.KNIGHT, chess.BLACK)) +
        len(board.pieces(chess.BISHOP, chess.WHITE)) + 
        len(board.pieces(chess.BISHOP, chess.BLACK))
    )
    return queens == 0 or (queens == 2 and minor_pieces <= 2)


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
