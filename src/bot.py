import chess
import math

# Piece-square tables for chess pieces
piece_square_tables: dict[chess.Piece, list[int]] = {
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
piece_values: dict[chess.Piece, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
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


# Minimax algorithm with alpha-beta pruning, transposition table
def minimax(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool, transposition_table: dict | None = None) -> tuple[float, chess.Move | None]:
    if transposition_table:
        position_hash = board.fen()
        if position_hash in transposition_table:
            stored_value, stored_depth, stored_move = transposition_table[position_hash]
            if stored_depth == depth:
                return stored_value, stored_move
    
    if depth == 0 or board.is_game_over():
        return evaluation(board), None
    
    best_value = -math.inf if is_maximizing else math.inf
    best_move = None

    moves = list(board.legal_moves)

    # Move ordering
    moves.sort(key=lambda move: (
        # Captures of higher value pieces
        1000 * (board.piece_at(move.to_square).piece_type 
                if board.piece_at(move.to_square) else 0)
        # Promotions
        + 100 * (1 if move.promotion else 0)
        # Checks
        + 10 * board.gives_check(move)
    ), reverse=True)

    for move in moves:
        board.push(move)
        value, _ = minimax(board, depth - 1, alpha, beta, not is_maximizing, transposition_table)
        board.pop()
        
        if is_maximizing:
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        else:
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, value)
            
        if beta <= alpha:
            break

    if transposition_table:
        transposition_table[position_hash] = (best_value, depth, best_move)

    return best_value, best_move


# Simple Evaluation Function
def evaluation(board: chess.Board) -> float:
    if board.is_checkmate():
        return -2147483648 if board.turn == chess.WHITE else 2147483647
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0

    score += evaluate_position(board)
    score += evaluate_material(board)

    return score


def evaluate_position(board: chess.Board) -> int:
    score = 0
    endgame = is_endgame(board)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = get_position_value(piece, square, endgame)
            score += value if piece.color == chess.WHITE else -value
    return score


def evaluate_material(board: chess.Board) -> int:
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score
