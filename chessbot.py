import chess
from modules.minimax import minimax
import math
import torch
from typing import Dict


class ChessBot:
    def __init__(self, max_depth: int, model: torch.nn.Module, transposition_table: Dict | None = None) -> None:
        self.max_depth = max_depth
        self.transposition_table = transposition_table
        self.model = model

        self.model.eval()
        self.model = self._optimize_model(model)

    def search(self, board: chess.Board) -> chess.Move:
        is_maximizing = True if board.turn == chess.WHITE else False
        _, best_move = minimax(board, self.max_depth, -math.inf, math.inf, is_maximizing, self.model, self.transposition_table)

        return best_move
    
    def _optimize_model(self, model: torch.nn.Module) -> torch.jit.ScriptModule:
        example_input = torch.zeros(1, 768)
        traced_model = torch.jit.trace(model, example_input)
        return traced_model
