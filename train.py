import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from typing import List

from modules.nnue import NNUE, ChessDataset
from modules.utils import load_config


def train(model: torch.nn.Module, optimizer, train_loader: torch.utils.data.DataLoader, num_epochs: int, device: str) -> List[float]:
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for i, (features, evaluations) in enumerate(progress_bar):
            features, evaluations = features.to(device), evaluations.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, evaluations)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        epoch_losses.append(running_loss / len(train_loader))

    # Saving the model
    torch.save(model.state_dict(), './models/nnue_model.pth')

    return epoch_losses


def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str) -> float:
    criterion = nn.MSELoss()

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

        for i, (features, evaluations) in enumerate(progress_bar):
            features, evaluations = features.to(device), evaluations.to(device)
            outputs = model(features)
            loss = criterion(outputs, evaluations)
            
            running_loss += loss.item()
            progress_bar.set_postfix(val_loss=running_loss / (i + 1))

    return running_loss / len(val_loader)


def main() -> None:
    torch.manual_seed(37)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = "config/default.yaml"
    config = load_config(config_path=config_path)

    model_save_path = config["paths"]["models"]
    data_folder_path = config["paths"]["data"]
    checkpoint_model = config["paths"]["checkpoint_model"]

    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    num_workers = config["training"]["num_workers"]

    os.makedirs(model_save_path, exist_ok=True)

    # Load the Data and split into training and validation sets
    data = ChessDataset(data_folder_path)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Training
    nnue = NNUE()
    optimizer = torch.optim.Adam(nnue.parameters(), lr=learning_rate)

    if checkpoint_model:
        nnue.load_state_dict(torch.load(checkpoint_model))

    nnue = nnue.to(device)

    train(nnue, optimizer, train_loader, num_epochs, device)

    # Validation
    validate(nnue, val_loader, device)


if __name__ == "__main__":
    main()
