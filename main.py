import chess
from src.minimax import minimax
import math

class ChessBot:
    def __init__(self, max_depth: int, transposition_table: dict | None = None) -> None:
        self.max_depth = max_depth
        self.transposition_table = transposition_table

    def search(self, board: chess.Board) -> chess.Move:
        is_maximizing = True if board.turn == chess.WHITE else False
        _, best_move = minimax(board, self.max_depth, -math.inf, math.inf, is_maximizing, self.transposition_table)

        return best_move


chessbot = ChessBot(4, None)

board = chess.Board()

board.push(chessbot.search(board))

print(board)
