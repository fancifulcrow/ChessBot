import chess
from modules.minimax import minimax
import math
import torch


class ChessBot:
    def __init__(self, max_depth: int, model: torch.nn.Module, transposition_table: dict | None = None) -> None:
        self.max_depth = max_depth
        self.transposition_table = transposition_table
        self.model = model

    def search(self, board: chess.Board) -> chess.Move:
        is_maximizing = True if board.turn == chess.WHITE else False
        _, best_move = minimax(board, self.max_depth, -math.inf, math.inf, is_maximizing, self.model, self.transposition_table)

        return best_move
