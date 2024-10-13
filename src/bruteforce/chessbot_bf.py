import chess
import math


class ChessBotBF:
    def __init__(self, max_depth:int) -> None:
        self.max_depth = max_depth


    # Minimax with Alpha-Beta Pruning
    def minimax(self, board:chess.Board, depth:int, alpha:float, beta:float, is_maximizing:bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if is_maximizing:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval


    def get_best_move(self, board:chess.Board) -> chess.Move:
        pass


    def evaluate_board(self, board: chess.Board) -> float:
        pass