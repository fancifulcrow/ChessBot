# Chess-Bot

## Overview

This project introduces chess engine capable of evaluating positions and making decisions based on neural network insights. The bot is designed with two core components: **Search** and **Evaluation**. While initially leveraging classical techniques, the project explores the integration of a neural network to improve the evaluation process, inspired by **NNUE** (Efficiently Updatable Neural Network). This NNUE implementation is simpler than the one used in Stockfish.

<p align="center">
  <img src="images/output.svg" alt="A Chessboard illustration">
</p>

## Search

The search component is built on the **Minimax algorithm**, a fundamental decision-making algorithm in game theory that maximizes the best possible outcome for one player while minimizing the potential for the other. In chess:

- **White** aims to select moves leading to the most favorable board positions.
- **Black** aims to choose moves that minimize White's advantage, effectively maximizing its own best outcome.

To enhance performance, a **transposition table** is utilized. This table stores previously evaluated positions, allowing the bot to avoid redundant calculations and thus speed up decision-making.

## Evaluation

Initially, the bot used a hand-crafted evaluation function. This function combined piece-square tables, which capture positional values for pieces on specific squares, with the material values of each piece. While this traditional approach offered solid evaluations, it was later replaced with a neural network approach inspired by **NNUE** principles.

The NNUE (Efficiently Updatable Neural Network) approach is designed to provide a fast evaluation of board positions through a compact neural network architecture. NNUE must meet the following conditions:

1. **Sparse Inputs**  
The input vector must contain minimal non-zero elements. For our model, the input layer size is set to **768**. Each position on the board is represented by a tuple, **(piece_square, piece_type, piece_color)**, yielding 768 inputs (64 squares × 6 piece types × 2 colors). Given that a chessboard has at most 32 pieces, only a small portion of the input layer is active at any given time.

2. **Minimal Change in Inputs**  
Inputs should vary as little as possible between successive evaluations. In our implementation, at most only four inputs might change (castling).

3. **Simplicity**   
The network architecture should be small and efficient to ensure performance on CPUs. Our neural network follows a minimal structure of **768 -> 8 -> 8 -> 1** neurons. So we only have two hidden layers with eight neurons each.

## Performance & Limitations
Currently, the bot has limitations in terms of computational performance:

- **Search Depth**: On a 12th Gen Intel(R) Core(TM) i5-12500H processor, the bot reaches a practical search depth of **4** in live games.
- **Model Optimization**: The NNUE model has not been quantized, which could further improve its efficiency.
- **Lack of Multithreading**: The bot currently operates in a single-threaded mode, which limits its processing speed.
- **Language Constraints**: Python, though convenient for development, introduces some inherent performance limitations. The project could be further optimized by exploring compiled languages like C.

These performance bottlenecks represent future avenues for optimization.

## Dataset
The NNUE is trained using the [Chess FENs + Evaluations dataset](https://www.kaggle.com/datasets/dev102/chess-fens-evaluations-dataset) on Kaggle.

## References
official-stockfish. "*nnue-pytorch.*" GitHub, https://github.com/official-stockfish/nnue-pytorch.

For further details, refer to the [Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)
