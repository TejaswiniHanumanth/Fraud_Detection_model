import torch
from torch.utils.data import DataLoader
from dataset import RetailDataset, get_transforms
from model import get_model
import yaml

def validate():
    # Load Configurations
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load Dataset
    test_dataset = RetailDataset(config["paths"]["test_csv"], 
                                  config["paths"]["test_frames"], 
                                  transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False, num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load(config["paths"]["model_save_path"]))
    model = model.to(device)
    model.eval()

    # Evaluate Model
    correct_cash = 0
    correct_invoice = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            label_cash = labels['cash_transaction'].to(device)
            label_invoice = labels['invoice_provided'].to(device)
            
            outputs = model(images)
            _, predicted_cash = torch.max(outputs[:, 0], 1)
            _, predicted_invoice = torch.max(outputs[:, 1], 1)
            
            correct_cash += (predicted_cash == label_cash).sum().item()
            correct_invoice += (predicted_invoice == label_invoice).sum().item()
            total += label_cash.size(0)

    print(f"Accuracy (Cash Transaction): {correct_cash / total}")
    print(f"Accuracy (Invoice Provided): {correct_invoice / total}")

if __name__ == "__main__":
    validate()
