import chess
import torch
import time

from modules.nnue import NNUE
from chessbot import ChessBot


model_path = "models/nnue_model_checkpoint.pth"


def main() -> None:
    transposition_table = dict()

    # Load the model
    nnue = NNUE()
    nnue.load_state_dict(torch.load(model_path, weights_only=True))
    nnue.eval()
    
    # Create the chessbot object
    chessbot = ChessBot(max_depth=5, model=nnue, transposition_table=transposition_table)

    # Create board and play
    board = chess.Board()

    while not board.is_game_over():
        start = time.time()

        move = chessbot.search(board)

        end = time.time()

        board.push(move)

        print("----------------------")
        print(board)
        print(f"Time Elapsed: {end - start:.2f}s")

    print(board.result())


if __name__ == "__main__":
    main()
