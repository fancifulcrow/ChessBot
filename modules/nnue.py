import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Tuple

from .utils import fen_to_rep


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # features are a tuple (piece_square, piece_type, piece_color): 64 * 6 * 2 = 768
        self.fc1 = nn.Linear(768, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.clipped_ReLU(x)
        x = self.fc2(x)
        x = self.clipped_ReLU(x)
        x = self.fc3(x)

        return x

    def clipped_ReLU(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0, max=127)


class ChessDataset(Dataset):
    def __init__(self, folder_path, chunk_size=10000) -> None:
        self.folder_path = folder_path
        self.data_files = [os.path.join(folder_path, file_name) 
                           for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_index = 0
        self.current_file_index = 0

        self.length = sum(sum(1 for _ in pd.read_csv(file, chunksize=self.chunk_size)) for file in self.data_files) * self.chunk_size

    # the dataset was too large for memory
    def _load_chunk(self) -> None:
        if self.current_file_index >= len(self.data_files):
            raise IndexError("No more data available.")

        if self.current_chunk is None or self.current_chunk_index >= len(self.current_chunk):
            file_path = self.data_files[self.current_file_index]
            self.current_chunk = pd.read_csv(file_path, chunksize=self.chunk_size)
            self.current_chunk_index = 0

            try:
                self.current_chunk = next(self.current_chunk)
            except StopIteration:
                self.current_file_index += 1
                self.current_chunk = None
                self._load_chunk()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        self._load_chunk()

        if idx >= len(self.current_chunk):
            self._load_chunk()
            idx = 0

        row = self.current_chunk.iloc[idx]
        feature_vector = fen_to_rep(row["FEN"])
        evaluation = torch.tensor(row["Evaluation"], dtype=torch.float32).unsqueeze(0)

        return feature_vector, evaluation
