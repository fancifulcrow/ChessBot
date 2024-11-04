import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

from modules.nnue import NNUE, ChessDataset


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("./models", exist_ok=True)

    # Load the Data and split into training and validation sets
    data = ChessDataset("./data")

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size

    train_data, val_data = random_split(data, [train_size, val_size])

    batch_size = 256

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Training
    nnue = NNUE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nnue.parameters(), lr=1e-3)

    nnue = nnue.to(device)

    num_epochs = 50

    for epoch in range(num_epochs):
        nnue.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for i, (features, evaluations) in progress_bar:
            features, evaluations = features.to(device), evaluations.to(device)
            
            optimizer.zero_grad()
            outputs = nnue(features)
            loss = criterion(outputs, evaluations)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
    
    # Validation
    nnue.eval()
    running_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), desc="Validation", unit="batch")
        for i, (features, evaluations) in progress_bar:
            features, evaluations = features.to(device), evaluations.to(device)
            outputs = nnue(features)
            loss = criterion(outputs, evaluations)
            
            running_loss += loss.item()
            progress_bar.set_postfix(val_loss=running_loss / (i + 1))

    # Saving the model
    torch.save(nnue.state_dict(), './models/nnue_model.pth')
