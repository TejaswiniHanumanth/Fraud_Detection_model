import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RetailDataset, get_transforms
from model import get_model
import yaml

def train():
    # Load Configurations
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load Dataset
    train_dataset = RetailDataset(config["paths"]["train_csv"], 
                                   config["paths"]["train_frames"], 
                                   transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=config["model"]["pretrained"], 
                      num_classes=config["model"]["num_classes"], 
                      dropout=config["model"]["dropout"])
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training Loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            label_cash = labels['cash_transaction'].to(device)
            label_invoice = labels['invoice_provided'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_cash = criterion(outputs[:, 0], label_cash)
            loss_invoice = criterion(outputs[:, 1], label_invoice)
            loss = loss_cash + loss_invoice
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Save the Model
    torch.save(model.state_dict(), config["paths"]["model_save_path"])
    print("Model saved!")

if __name__ == "__main__":
    train()
