import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import TransactionDataset
from model import TransactionModel
from torchvision import transforms
from load_config import load_config

config = load_config()

# Paths and hyperparameters
train_videos = config['paths']['train_videos']
labels = config['labels']
batch_size = config['hyperparameters']['batch_size']
epochs = config['hyperparameters']['epochs']
learning_rate = config['hyperparameters']['learning_rate']

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = TransactionDataset(train_videos, labels, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = TransactionModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for frames, labels in train_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), config['paths']['model_save_path'])
