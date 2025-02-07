import chess
import math
import torch
from typing import Dict, Tuple

from .utils import get_position_value, piece_values, is_endgame, board_to_rep


# Minimax algorithm with alpha-beta pruning, transposition table
def minimax(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool, model: torch.nn.Module, transposition_table: Dict | None = None) -> Tuple[float, chess.Move | None]:
    if transposition_table:
        position_hash = board.fen()
        if position_hash in transposition_table:
            stored_value, stored_depth, stored_move = transposition_table[position_hash]
            if stored_depth == depth:
                return stored_value, stored_move
    
    if board.is_game_over() or depth == 0:
        return evaluation(board, model), None
    
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

    if transposition_table:
        transposition_table[position_hash] = (best_value, depth, best_move)

    return best_value, best_move


def evaluation(board: chess.Board, model: torch.nn.Module) -> float:
    if board.is_checkmate():
        return -2147483648 if board.turn == chess.WHITE else 2147483647
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    inputs = board_to_rep(board)

    return model(inputs).item()


# Simple Evaluation Function. This is the implementation of a basic handcrafted evaluation function
def simple_evaluation(board: chess.Board) -> float:
    if board.is_checkmate():
        return -2147483648 if board.turn == chess.WHITE else 2147483647
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0

    score += evaluate_position(board)
    score += evaluate_material(board)

    return score


def evaluate_position(board: chess.Board) -> float:
    score = 0
    endgame = is_endgame(board)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = get_position_value(piece, square, endgame)
            score += value if piece.color == chess.WHITE else -value
    return score


def evaluate_material(board: chess.Board) -> float:
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score


def quiescence_search(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
    # Implement later. Maybe...
    ...
