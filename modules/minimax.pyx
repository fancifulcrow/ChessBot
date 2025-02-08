# cython: language_level=3
import chess
import torch
from typing import Tuple
cimport cython
from cpython.ref cimport PyObject
from .utils import piece_to_index

# Type definitions
ctypedef float value_type
ctypedef object move_type
ctypedef dict trans_table_type

@cython.boundscheck(False)
@cython.wraparound(False)
def board_to_rep(board) -> torch.Tensor:
    cdef:
        int square, piece_offset, piece_type, color_offset, feature_index
        object piece
    
    features = torch.zeros(768, dtype=torch.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_offset = piece_to_index[piece.symbol()]
            piece_type = piece_offset % 6
            color_offset = piece_offset // 6
            feature_index = square * 12 + color_offset * 6 + piece_type
            features[feature_index] = 1.0

    return features

@cython.boundscheck(False)
@cython.wraparound(False)
def minimax(board, int depth, double alpha, double beta, bint is_maximizing, 
            model, transposition_table=None) -> Tuple[float, object]:
    cdef:
        double best_value, value
        str position_hash
        list moves
        object move, best_move
        int capture_value
    
    if transposition_table is not None:
        position_hash = board.fen()
        if position_hash in transposition_table:
            stored_value, stored_depth, stored_move = transposition_table[position_hash]
            if stored_depth == depth:
                return stored_value, stored_move
    
    if board.is_game_over() or depth == 0:
        return evaluation(board, model), None
    
    best_value = -float('inf') if is_maximizing else float('inf')
    best_move = None

    moves = list(board.legal_moves)
    
    # Move ordering
    moves.sort(key=lambda move: (
        1000 * (board.piece_at(move.to_square).piece_type 
                if board.piece_at(move.to_square) else 0)
        + 100 * (1 if move.promotion else 0)
        + 10 * board.gives_check(move)
    ), reverse=True)

    for move in moves:
        board.push(move)
        value, _ = minimax(board, depth - 1, alpha, beta, not is_maximizing, model, transposition_table)
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

    if transposition_table is not None:
        transposition_table[position_hash] = (best_value, depth, best_move)

    return best_value, best_move

@cython.boundscheck(False)
@cython.wraparound(False)
def evaluation(board, model) -> float:
    if board.is_checkmate():
        return -2147483648.0 if board.turn == chess.WHITE else 2147483647.0
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    inputs = board_to_rep(board)
    return model(inputs).item()
