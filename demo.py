import chess
import torch

from modules.nnue import NNUE
from main import ChessBot

model_path = "models/nnue_model.pth"

if __name__ == "__main__":
    transposition_table = dict()

    # Load the model
    nnue = NNUE()
    nnue.load_state_dict(torch.load(model_path, weights_only=True))
    nnue.eval()
    
    # Create the chessbot object
    chessbot = ChessBot(max_depth=4, model=nnue, transposition_table=transposition_table)

    # Create board and play
    board = chess.Board()

    move = chessbot.search(board)

    board.push(move)

    print(board)
